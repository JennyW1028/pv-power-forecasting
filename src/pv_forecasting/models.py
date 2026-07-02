from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_output))


class InformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, e_layers: int, seq_len: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.position_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder_layers = nn.ModuleList(
            [InformerEncoderLayer(d_model, n_heads, d_ff or d_model * 4, dropout) for _ in range(e_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.position_encoding(x))
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int = 64):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self) -> torch.Tensor:
        adj = torch.mm(self.node_emb, self.node_emb.t())
        adj = F.relu(adj)
        return F.softmax(adj, dim=1)


class DynamicAdjacency(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)
        key = self.key_proj(x)
        sim = torch.bmm(query, key.transpose(1, 2))
        sim = sim / (self.temperature.abs() + 1e-8)
        return F.softmax(sim, dim=-1)


class DenseGCN(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        degree = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag_embed(d_inv_sqrt)
        adj_norm = torch.matmul(torch.matmul(d_mat, adj), d_mat)
        return torch.matmul(adj_norm, self.linear(x))


class DenseGAT(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, concat: bool = False, dropout: float = 0.2, alpha: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.out_features = out_features
        self.dropout = dropout
        self.proj = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.attn = nn.Parameter(torch.zeros(num_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.attn)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if adj.size(0) == 1 and batch_size > 1:
            adj = adj.expand(batch_size, -1, -1)

        projected = self.proj(x).view(batch_size, num_nodes, self.num_heads, self.out_features)
        left = projected.unsqueeze(2)
        right = projected.unsqueeze(1)
        pair_features = torch.cat(
            [
                left.expand(-1, -1, num_nodes, -1, -1),
                right.expand(-1, num_nodes, -1, -1, -1),
            ],
            dim=-1,
        )
        attn = self.attn.view(1, 1, 1, self.num_heads, 2 * self.out_features)
        scores = self.leaky_relu((pair_features * attn).sum(dim=-1))
        scores = scores.masked_fill((adj > 0).unsqueeze(-1) == 0, float("-inf"))
        weights = F.softmax(scores, dim=2)
        weights = F.dropout(weights, self.dropout, training=self.training)

        weights = weights.permute(0, 3, 1, 2)
        projected = projected.permute(0, 2, 1, 3)
        out = torch.matmul(weights, projected)
        if self.concat:
            return out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
        return out.mean(dim=1)


class GCNInformer(nn.Module):
    def __init__(self, num_nodes: int, input_features: int, hidden_dim: int = 32, seq_len: int = 24, num_gcn_layers: int = 2, num_encoder_layers: int = 1, num_heads: int = 4, embed_dim: int = 64):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, embed_dim)
        layers = [DenseGCN(input_features, hidden_dim)]
        layers.extend(DenseGCN(hidden_dim, hidden_dim) for _ in range(num_gcn_layers - 1))
        self.gcn_layers = nn.ModuleList(layers)
        self.informer = InformerEncoder(hidden_dim, num_heads, num_encoder_layers, seq_len)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        adj = self.adaptive_adj()
        outputs = []
        for step in range(x.size(1)):
            x_t = x[:, step, :, :]
            for layer_idx, layer in enumerate(self.gcn_layers):
                x_t = layer(x_t, adj)
                if layer_idx != len(self.gcn_layers) - 1:
                    x_t = self.dropout(F.relu(x_t))
            outputs.append(x_t)
        graph_seq = torch.stack(outputs, dim=1).sum(dim=2)
        encoded = self.informer(graph_seq)
        out = self.dropout(F.relu(self.fc1(encoded[:, -1, :])))
        return self.fc2(out)


class GATInformer(nn.Module):
    def __init__(self, num_nodes: int, input_features: int, hidden_dim: int = 32, seq_len: int = 24, num_gat_layers: int = 1, num_encoder_layers: int = 1, num_heads: int = 4, static_embed_dim: int = 32, dynamic_hidden_dim: int = 16, gat_dropout: float = 0.2):
        super().__init__()
        self.static_adj = AdaptiveAdjacency(num_nodes, static_embed_dim)
        self.dynamic_adj = DynamicAdjacency(input_features, dynamic_hidden_dim)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        layers = [DenseGAT(input_features, hidden_dim, num_heads=num_heads, concat=False, dropout=gat_dropout)]
        layers.extend(DenseGAT(hidden_dim, hidden_dim, num_heads=num_heads, concat=False, dropout=gat_dropout) for _ in range(num_gat_layers - 1))
        self.gat_layers = nn.ModuleList(layers)
        self.informer = InformerEncoder(hidden_dim, num_heads, num_encoder_layers, seq_len)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        static_adj = self.static_adj().unsqueeze(0)
        alpha = torch.sigmoid(self.fusion_alpha)
        outputs = []
        for step in range(x.size(1)):
            x_t = x[:, step, :, :]
            dynamic_adj = self.dynamic_adj(x_t)
            adj = alpha * static_adj + (1.0 - alpha) * dynamic_adj
            for layer_idx, layer in enumerate(self.gat_layers):
                x_t = layer(x_t, adj)
                if layer_idx != len(self.gat_layers) - 1:
                    x_t = self.dropout(F.relu(x_t))
            outputs.append(x_t)
        graph_seq = torch.stack(outputs, dim=1).sum(dim=2)
        encoded = self.informer(graph_seq)
        out = self.dropout(F.relu(self.fc1(encoded[:, -1, :])))
        return self.fc2(out)


class GCNLSTM(nn.Module):
    def __init__(self, num_nodes: int, input_features: int, hidden_dim: int = 32, seq_len: int = 24, num_gcn_layers: int = 2, num_lstm_layers: int = 1, embed_dim: int = 64, lstm_dropout: float = 0.0):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, embed_dim)
        layers = [DenseGCN(input_features, hidden_dim)]
        layers.extend(DenseGCN(hidden_dim, hidden_dim) for _ in range(num_gcn_layers - 1))
        self.gcn_layers = nn.ModuleList(layers)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_lstm_layers, batch_first=True, dropout=lstm_dropout if num_lstm_layers > 1 else 0.0)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        adj = self.adaptive_adj()
        outputs = []
        for step in range(x.size(1)):
            x_t = x[:, step, :, :]
            for layer_idx, layer in enumerate(self.gcn_layers):
                x_t = layer(x_t, adj)
                if layer_idx != len(self.gcn_layers) - 1:
                    x_t = self.dropout(F.relu(x_t))
            outputs.append(x_t)
        graph_seq = torch.stack(outputs, dim=1).sum(dim=2)
        lstm_out, _ = self.lstm(graph_seq)
        out = self.dropout(F.relu(self.fc1(lstm_out[:, -1, :])))
        return self.fc2(out)


class QuantileLSTM(nn.Module):
    def __init__(self, input_features: int, seq_len: int = 24, d_model: int = 128, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1, num_quantiles: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.value_embedding = nn.Linear(input_features, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.lstm_dropout = nn.Dropout(dropout)
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_size * seq_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.quantile_outputs = nn.ModuleList(
            [nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)) for _ in range(num_quantiles)]
        )
        self._init_lstm_weights()

    def _init_lstm_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.input_dropout(self.value_embedding(x))
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        features = self.shared_layers(self.lstm_dropout(lstm_out).reshape(batch_size, -1))
        return torch.cat([head(features) for head in self.quantile_outputs], dim=1)


class QuantileInformer(nn.Module):
    def __init__(self, input_features: int, seq_len: int = 24, d_model: int = 128, n_heads: int = 4, e_layers: int = 2, dropout: float = 0.2, num_quantiles: int = 3):
        super().__init__()
        self.value_embedding = nn.Linear(input_features, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(e_layers)
            ]
        )
        self.shared_layers = nn.Sequential(
            nn.Linear(d_model * seq_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.quantile_outputs = nn.ModuleList(
            [nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)) for _ in range(num_quantiles)]
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding
        for layer in self.encoder_layers:
            x = layer(x)
        features = self.shared_layers(x.reshape(x.shape[0], -1))
        return torch.cat([head(features) for head in self.quantile_outputs], dim=1)


def build_model(model_name: str, input_features: int, seq_len: int, num_nodes: int = 7, hidden_dim: int = 32) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "gcn-informer":
        return GCNInformer(num_nodes=num_nodes, input_features=input_features, hidden_dim=hidden_dim, seq_len=seq_len)
    if model_name == "gat-informer":
        return GATInformer(num_nodes=num_nodes, input_features=input_features, hidden_dim=hidden_dim, seq_len=seq_len)
    if model_name == "gcn-lstm":
        return GCNLSTM(num_nodes=num_nodes, input_features=input_features, hidden_dim=hidden_dim, seq_len=seq_len)
    if model_name == "informer":
        return QuantileInformer(input_features=input_features, seq_len=seq_len)
    if model_name == "lstm":
        return QuantileLSTM(input_features=input_features, seq_len=seq_len)
    raise ValueError(f"Unsupported model: {model_name}")
