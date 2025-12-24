import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from torch.backends.cudnn import deterministic
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Optional, Literal, Tuple, List, Dict
from dataclasses import dataclass

from scipy.stats import norm


def simple_dst_fix(df: pd.DataFrame, start_at_midnight: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    df = df[~df["ds"].duplicated(keep="first")]

    start = df["ds"].iloc[0]
    if start_at_midnight:
        start = start.normalize()
    end = df["ds"].iloc[-1]
    full_idx = pd.date_range(start, end, freq="h")

    out = df.set_index("ds").reindex(full_idx)

    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].ffill()

    if out[num_cols].isna().any().any():
        out[num_cols] = out[num_cols].bfill()

    out = out.rename_axis("ds").reset_index()

    return out



def build_model_dp(model_cls, *args, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls(*args, **kwargs).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[DP] Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model

def save_state(model, path):
    sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(sd, path)

def load_state(model, path, map_location="cpu"):
    sd = torch.load(path, map_location=map_location)
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(sd)

def append_fourier_features(df_wide, K, period_days,
                            mode="vector",
                            start_idx=None,
                            start_day = None):
    if K <= 0:
        return df_wide, 0

    existing = [c for c in df_wide.columns if c.startswith("fourier_")]
    if start_idx is None:
        start_idx = len(existing)
    t0 = pd.to_datetime(df_wide["day"]).astype("int64") // 86_400_000_000_000
    if start_day is None:
        start_day = t0.min()
    t_day = (t0 - start_day).astype(float).to_numpy()
    P = float(period_days)

    N = len(df_wide);
    H = len([c for c in df_wide.columns if c.startswith("y_") and c.endswith("_s0")])
    if H == 0:
        H = len([c for c in df_wide.columns if c.startswith("y_")])

    if mode == "vector":
        feats = np.empty((N, 2 * K), dtype=np.float64)
        for k in range(1, K + 1):
            w = 2 * np.pi * k / P
            feats[:, 2 * (k - 1)] = np.sin(w * t_day)
            feats[:, 2 * (k - 1) + 1] = np.cos(w * t_day)
        cols = [f"fourier_{start_idx + j}" for j in range(feats.shape[1])]
        add = pd.DataFrame(feats, columns=cols, index=df_wide.index)
        return pd.concat([df_wide, add], axis=1), feats.shape[1]

    elif mode == "matrix":
        t_mat = t_day[:, None] + np.arange(H)[None, :] / float(H)
        S = np.stack([np.sin(2 * np.pi * k / P * t_mat) for k in range(1, K + 1)], axis=0)
        C = np.stack([np.cos(2 * np.pi * k / P * t_mat) for k in range(1, K + 1)], axis=0)

        blocks = []
        for h in range(H):
            blk = np.empty((N, 2 * K), dtype=np.float64)
            blk[:, 0::2] = S[:, :, h].T
            blk[:, 1::2] = C[:, :, h].T
            blocks.append(blk)
        Xf = np.concatenate(blocks, axis=1)
        cols = [f"fourier_{start_idx + j}" for j in range(Xf.shape[1])]
        add = pd.DataFrame(Xf, columns=cols, index=df_wide.index)
        return pd.concat([df_wide, add], axis=1), Xf.shape[1]

    else:
        raise ValueError("mode must be 'vector' or 'matrix'")

def custom_collate_fn(batch):
    x_shared_list = [item[0] for item in batch]
    x_indep_list = [item[1] for item in batch]
    x_text_list = [item[2] for item in batch]
    xf_list = [item[3] for item in batch]
    y_list = [item[4] for item in batch]

    # Stack x_shared
    if x_shared_list[0] is not None:
        x_shared = torch.stack(x_shared_list)
    else:
        x_shared = None

    # Stack x_indep per series
    n_series = len(x_indep_list[0])
    x_indep = [torch.stack([x_indep_list[b][s] for b in range(len(batch))]) for s in range(n_series)]

    # Stack x_text per series per embedding
    # x_text[s][e] = [B, T, dim_e] or None
    x_text = []
    for s in range(n_series):
        n_emb = len(x_text_list[0][s])
        series_text = []
        for e in range(n_emb):
            if x_text_list[0][s][e] is not None:
                series_text.append(torch.stack([x_text_list[b][s][e] for b in range(len(batch))]))
            else:
                series_text.append(None)
        x_text.append(series_text)

    # Stack xf
    if xf_list[0] is not None:
        xf = torch.stack(xf_list)
    else:
        xf = None

    # Stack y per series
    y = [torch.stack([y_list[b][s] for b in range(len(batch))]) for s in range(n_series)]

    return x_shared, x_indep, x_text, xf, y


class FeatureAttention(nn.Module):
    def __init__(self, latent_dim, d_model=128, nhead=1, dropout=0.0, use_gate=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_gate = use_gate

        self.feat_embed = nn.Parameter(torch.randn(latent_dim, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

        if use_gate:
            self.gate = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Sigmoid())

    def forward(self, z):
        # z: [B, T, latent_dim]
        B, T, L = z.shape
        z_flat = z.reshape(B * T, L)

        tokens = z_flat.unsqueeze(-1) * self.feat_embed.unsqueeze(0)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)
        tokens = self.ln2(tokens + self.ffn(tokens))
        delta = self.out_proj(tokens).squeeze(-1)

        delta = delta.reshape(B, T, L)

        if self.use_gate:
            return z + self.gate(z) * delta
        else:
            return z + delta


class CrossSeriesAttention(nn.Module):
    def __init__(self, latent_dim, nhead=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, nhead, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, z):
        # z: [B, T, n_series, latent_dim]
        B, T, S, L = z.shape
        z_flat = z.view(B * T, S, L)

        attn_out, _ = self.attn(z_flat, z_flat, z_flat)
        z_flat = self.ln(z_flat + attn_out)

        return z_flat.view(B, T, S, L)


class FourierLayer(nn.Module):
    def __init__(self, fourier_dim, latent_dim, mode="vector", H=24, unique_alpha=False):
        super().__init__()
        self.F = fourier_dim
        self.mode = mode
        self.H = H
        self.unique_alpha = unique_alpha

        if mode == "vector":
            self.V = nn.Linear(1, latent_dim, bias=True)
            self.a = nn.Parameter(torch.randn(fourier_dim) * 0.01)
        else:
            if unique_alpha:
                self.V = nn.Linear(H, latent_dim, bias=True)
                self.a = nn.Parameter(torch.randn(fourier_dim) * 0.01)
            else:
                self.V = nn.Linear(H, latent_dim, bias=True)
                self.a = nn.Parameter(torch.randn(H, fourier_dim) * 0.01)

    def forward(self, xf):
        # xf: [B, T, fourier_dim] or [B, T, H * fourier_dim]
        B, T, _ = xf.shape

        if self.mode == "vector":
            fourier_weighted = xf * self.a.unsqueeze(0).unsqueeze(0)
            s = fourier_weighted.sum(-1, keepdim=True)
            fourier_contrib = self.V(s).squeeze(-2)
        else:
            if self.unique_alpha:
                xf_mat = xf.view(B, T, self.H, self.F)
                fourier_weighted = xf_mat * self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                s = fourier_weighted.sum(-1)
                fourier_contrib = self.V(s)
            else:
                xf_mat = xf.view(B, T, self.H, self.F)
                fourier_weighted = xf_mat * self.a.unsqueeze(0).unsqueeze(0)
                s = fourier_weighted.sum(-1)
                fourier_contrib = self.V(s)

        return fourier_contrib

    def reg_loss(self, lambdaf=0.0, harmonic_orders=None):
        device = self.a.device
        dtype = self.a.dtype

        loss = torch.tensor(0.0, device=device)

        if lambdaf > 0:
            if harmonic_orders is None:
                loss = loss + lambdaf * (self.a ** 2).sum()
            else:
                w = torch.as_tensor(harmonic_orders, dtype=dtype, device=device)
                if self.unique_alpha:
                    loss = loss + lambdaf * ((self.a ** 2) * (w ** 2)).sum()
                else:
                    loss = loss + lambdaf * ((self.a ** 2) * (w ** 2).unsqueeze(0)).sum()

        return loss


class OutputLayer(nn.Module):
    def __init__(self, H, latent_dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(H, latent_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(H))

    def forward(self, z):
        # z: [B, T, latent_dim]
        # o: [B, T, H]
        o = z @ self.A.T + self.c
        return o


class RNN_fourier(nn.Module):

    def __init__(
            self,
            fourier_dim=0,
            xf_mode="vector",
            latent_dim=24,
            d_model=128,
            nhead=4,
            activation="relu",
            learn_z0=True,
            dropout=0.0,
            H=24,
            use_gate=True,
            nonneg_U0=False,
            rnn_type="rnn",
            unique_alpha=False,
            n_series = 1,
            n_shared=0,
            n_indep=None,
            embed_dim=32,
            embed_hidden=64,
            text_embed_dims=None,
            text_embed_hidden=256,
    ):
        super().__init__()

        self.H = H
        self.latent_dim = latent_dim
        self.F = fourier_dim
        self.mode = xf_mode
        self.nonneg_U0 = nonneg_U0
        self.unique_alpha = unique_alpha
        self.n_series = n_series
        self.n_shared = n_shared
        self.n_indep = n_indep


        if self.n_indep is None:
            self.n_indep = [1] * n_series

        self.embed_dim = embed_dim
        self.n_features = self.n_indep[0]

        self.text_embed_dims = text_embed_dims if text_embed_dims is not None else []
        self.n_text_features = len(self.text_embed_dims)

        if self.n_text_features > 0:
            self.text_embed = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, text_embed_hidden),
                    nn.ReLU(),
                    nn.Linear(text_embed_hidden, text_embed_hidden),
                    nn.ReLU(),
                    nn.Linear(text_embed_hidden, embed_dim),
                )
                for dim in self.text_embed_dims
            ])
        else:
            self.text_embed = None

        # Shared RNN
        if n_shared > 0:
            if rnn_type == "rnn":
                self.rnn_shared = nn.RNN(
                    input_size=n_shared,
                    hidden_size=latent_dim,
                    num_layers=1,
                    batch_first=True,
                    nonlinearity='relu' if activation == 'relu' else 'tanh'
                )
            else:
                self.rnn_shared = nn.GRU(
                    input_size=n_shared,
                    hidden_size=latent_dim,
                    num_layers=1,
                    batch_first=True
                )
        else:
            self.rnn_shared = None

        # Shared MLP embedding per feature type
        if any(n > 0 for n in self.n_indep):
            self.feature_embed = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, embed_hidden),
                    nn.ReLU(),
                    nn.Linear(embed_hidden, embed_hidden),
                    nn.ReLU(),
                    nn.Linear(embed_hidden, embed_dim),
                )
                for _ in range(self.n_features)
            ])

            # Independent RNN per series (input is concatenated embeddings)
            rnn_input_size = self.n_features * embed_dim + self.n_text_features * embed_dim

            if rnn_type == "rnn":
                self.rnn_indep = nn.ModuleList([
                    nn.RNN(
                        input_size=rnn_input_size,
                        hidden_size=latent_dim,
                        num_layers=1,
                        batch_first=True,
                        nonlinearity='relu' if activation == 'relu' else 'tanh'
                    )
                    for _ in range(n_series)
                ])
            else:
                self.rnn_indep = nn.ModuleList([
                    nn.GRU(
                        input_size=rnn_input_size,
                        hidden_size=latent_dim,
                        num_layers=1,
                        batch_first=True
                    )
                    for _ in range(n_series)
                ])
        else:
            self.feature_embed = None
            self.rnn_indep = None

        self.learn_z0 = learn_z0
        if learn_z0:
            # Shared RNN initial state
            if n_shared > 0:
                self.z0_shared = nn.Parameter(torch.zeros(1, 1, latent_dim))

            # Independent RNN initial states (per series)
            if any(n > 0 for n in self.n_indep):
                self.z0 = nn.ParameterList([
                    nn.Parameter(torch.zeros(1, 1, latent_dim))
                    for _ in range(n_series)
                ])

        if fourier_dim > 0:
            self.fourier_layers = nn.ModuleList([
                FourierLayer(fourier_dim, latent_dim, mode=xf_mode, H=H, unique_alpha=unique_alpha)
                for _ in range(n_series)
            ])
        else:
            self.fourier_layers = None

        self.feature_attn_layers = nn.ModuleList([
            FeatureAttention(latent_dim, d_model, nhead=1, dropout=dropout, use_gate=use_gate)
            for _ in range(n_series)
        ])

        if n_series > 1:
            self.cross_attn = CrossSeriesAttention(latent_dim, nhead=4, dropout=dropout)
        else:
            self.cross_attn = None

        self.output_layers = nn.ModuleList([
            OutputLayer(H, latent_dim)
            for _ in range(n_series)
        ])

    def reg_loss(self, lambda0=0.0, lambdaf=0.0, harmonic_orders=None):
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.fourier_layers is not None:
            for s in range(self.n_series):
                loss = loss + self.fourier_layers[s].reg_loss(lambdaf=lambdaf, harmonic_orders=harmonic_orders)
        return loss

    def forward(self, x_shared, x_indep, x_text, xf, z0=None):

        B, T = x_shared.shape[:2] if x_shared is not None else x_indep[0].shape[:2]

        # Shared RNN
        if self.rnn_shared is not None and x_shared is not None:
            if z0 is None and self.learn_z0:
                h0_shared = self.z0_shared.expand(-1, B, -1).contiguous()
            else:
                h0_shared = z0
            h_shared, _ = self.rnn_shared(x_shared, h0_shared)
        else:
            h_shared = 0

        # Independent RNN per series
        h_indep = []
        if self.rnn_indep is not None and x_indep is not None:
            for s in range(self.n_series):
                x_s = x_indep[s]  # [B, T, n_features]

                # Embed general features with shared MLP
                embedded_features = []
                for f in range(self.n_features):
                    x_f = x_s[:, :, f:f + 1]  # [B, T, 1]
                    emb_f = self.feature_embed[f](x_f)  # [B, T, embed_dim]
                    embedded_features.append(emb_f)

                # Embed text features with separate MLPs
                if self.text_embed is not None and x_text is not None:
                    for e in range(self.n_text_features):
                        if x_text[s][e] is not None:
                            x_t = x_text[s][e]  # [B, T, text_embed_dims[e]]
                            emb_t = self.text_embed[e](x_t)  # [B, T, embed_dim]
                            embedded_features.append(emb_t)

                # Concatenate all embeddings
                x_embedded = torch.cat(embedded_features, dim=-1)

                # Series-specific initial state
                if z0 is None and self.learn_z0:
                    h0_s = self.z0[s].expand(-1, B, -1).contiguous()
                else:
                    h0_s = z0

                # Series-specific RNN
                h_s, _ = self.rnn_indep[s](x_embedded, h0_s)
                h_indep.append(h_s)
        else:
            device = x_shared.device if x_shared is not None else x_indep[0].device
            h_indep = [torch.zeros(B, T, self.latent_dim, device=device) for _ in range(self.n_series)]

        # Fourier per series
        if self.fourier_layers is not None:
            fourier_contribs = [self.fourier_layers[s](xf) for s in range(self.n_series)]
        else:
            fourier_contribs = [0] * self.n_series

        # Combine: h_shared + h_indep + fourier per series
        z_list = []
        for s in range(self.n_series):
            z_s = h_shared + h_indep[s] + fourier_contribs[s]  # [B, T, latent_dim]
            z_list.append(z_s)

        # Feature attention per series
        z_attended = []
        for s in range(self.n_series):
            z_attended.append(self.feature_attn_layers[s](z_list[s]))
        z = torch.stack(z_attended, dim=2)  # [B, T, n_series, latent_dim]

        # Cross series attention
        if self.cross_attn is not None:
            z = self.cross_attn(z)

        # Output per series
        outputs = []
        for s in range(self.n_series):
            outputs.append(self.output_layers[s](z[:, :, s, :]))
        o = torch.stack(outputs, dim=2)  # [B, T, n_series, H]

        return o, z

    @staticmethod
    def extract_XY(df_wide, n_series, text_embed_dims=None):
        # Shared features: share_0, share_1, ...
        shared_cols = sorted([c for c in df_wide.columns if c.startswith("share_")],
                             key=lambda s: int(s.split("_")[1]))

        # Independent features: x_0_s0, x_1_s0, x_0_s1, ...
        indep_cols = [
            sorted([c for c in df_wide.columns if c.startswith("x_") and c.endswith(f"_s{s}")],
                   key=lambda c: int(c.split("_")[1]))
            for s in range(n_series)
        ]

        # Text embeddings: emb_{emb_idx}_s{series}_d{dim}
        # Group by series, then by embedding index
        n_text = len(text_embed_dims) if text_embed_dims is not None else 0
        text_cols = []
        for s in range(n_series):
            series_text_cols = []
            for e in range(n_text):
                # Get all dims for this embedding: emb_{e}_s{s}_d*
                emb_cols = sorted(
                    [c for c in df_wide.columns if c.startswith(f"emb_{e}_s{s}_d")],
                    key=lambda c: int(c.split("_d")[1])
                )
                series_text_cols.append(emb_cols)
            text_cols.append(series_text_cols)

        # Fourier: fourier_0, fourier_1, ...
        fourier_cols = sorted([c for c in df_wide.columns if c.startswith("fourier_")],
                              key=lambda s: int(s.split("_")[1]))

        # Targets: y_0_s0, y_1_s0, ..., y_0_s1, ...
        y_cols = [
            sorted([c for c in df_wide.columns if c.startswith("y_") and c.endswith(f"_s{s}")],
                   key=lambda c: int(c.split("_")[1]))
            for s in range(n_series)
        ]

        # Extract arrays
        X_shared = df_wide[shared_cols].to_numpy(dtype=np.float32) if shared_cols else None
        X_indep = [df_wide[indep_cols[s]].to_numpy(dtype=np.float32) for s in range(n_series)]

        # X_text[s][e] = array of shape [N, dim_e]
        X_text = []
        for s in range(n_series):
            series_text = []
            for e in range(n_text):
                if text_cols[s][e]:
                    series_text.append(df_wide[text_cols[s][e]].to_numpy(dtype=np.float32))
                else:
                    series_text.append(None)
            X_text.append(series_text)

        Xf = df_wide[fourier_cols].to_numpy(dtype=np.float32) if fourier_cols else None
        Y = [df_wide[y_cols[s]].to_numpy(dtype=np.float32) for s in range(n_series)]

        F_total = len(fourier_cols)
        K = F_total // 2 if F_total > 0 else 0
        harmonic_orders = sum(([k, k] for k in range(1, K + 1)), [])

        return X_shared, X_indep, X_text, Xf, Y, harmonic_orders

    @staticmethod
    def make_seq(X_shared, X_indep, X_text, Xf, Y, T=32, stride=1):
        N = len(Y[0])
        idx = list(range(0, N - T + 1, stride))

        X_shared_seq = np.stack([X_shared[i:i + T] for i in idx], axis=0).astype(
            np.float32) if X_shared is not None else None
        X_indep_seq = [np.stack([X_indep[s][i:i + T] for i in idx], axis=0).astype(np.float32)
                       for s in range(len(X_indep))]

        # X_text_seq[s][e] = [n_seq, T, dim_e]
        X_text_seq = []
        for s in range(len(X_text)):
            series_text_seq = []
            for e in range(len(X_text[s])):
                if X_text[s][e] is not None:
                    series_text_seq.append(
                        np.stack([X_text[s][e][i:i + T] for i in idx], axis=0).astype(np.float32)
                    )
                else:
                    series_text_seq.append(None)
            X_text_seq.append(series_text_seq)

        Xf_seq = np.stack([Xf[i:i + T] for i in idx], axis=0).astype(np.float32) if Xf is not None else None
        Y_seq = [np.stack([Y[s][i:i + T] for i in idx], axis=0).astype(np.float32) for s in range(len(Y))]

        return X_shared_seq, X_indep_seq, X_text_seq, Xf_seq, Y_seq

    class XYSeqDataset(Dataset):
        def __init__(self, X_shared_seq, X_indep_seq, X_text_seq, Xf_seq, Y_seq):
            self.X_shared = torch.from_numpy(X_shared_seq).float() if X_shared_seq is not None else None
            self.X_indep = [torch.from_numpy(x).float() for x in X_indep_seq]

            # X_text[s][e] = tensor or None
            self.X_text = []
            for s in range(len(X_text_seq)):
                series_text = []
                for e in range(len(X_text_seq[s])):
                    if X_text_seq[s][e] is not None:
                        series_text.append(torch.from_numpy(X_text_seq[s][e]).float())
                    else:
                        series_text.append(None)
                self.X_text.append(series_text)

            self.Xf = torch.from_numpy(Xf_seq).float() if Xf_seq is not None else None
            self.Y = [torch.from_numpy(y).float() for y in Y_seq]

        def __len__(self):
            return self.Y[0].shape[0]

        def __getitem__(self, i):
            x_shared = self.X_shared[i] if self.X_shared is not None else None
            x_indep = [self.X_indep[s][i] for s in range(len(self.X_indep))]

            # x_text[s][e] = [T, dim_e] or None
            x_text = []
            for s in range(len(self.X_text)):
                series_text = []
                for e in range(len(self.X_text[s])):
                    if self.X_text[s][e] is not None:
                        series_text.append(self.X_text[s][e][i])
                    else:
                        series_text.append(None)
                x_text.append(series_text)

            xf = self.Xf[i] if self.Xf is not None else None
            y = [self.Y[s][i] for s in range(len(self.Y))]
            return x_shared, x_indep, x_text, xf, y

    def get_dataloader(self, df_wide, fourier_config, test=False, start_day=None):
        df_wide = df_wide.sort_values("day")

        # Add Fourier features
        df_wide, _ = append_fourier_features(df_wide, K=fourier_config.K_weekly,
                                             period_days=fourier_config.P_WEEK, mode=fourier_config.mode,
                                             start_day=start_day)
        df_wide, _ = append_fourier_features(df_wide, K=fourier_config.K_monthly,
                                             period_days=fourier_config.P_MONTH, mode=fourier_config.mode,
                                             start_day=start_day)
        df_wide, _ = append_fourier_features(df_wide, K=fourier_config.K_yearly,
                                             period_days=fourier_config.P_yearly, mode=fourier_config.mode,
                                             start_day=start_day)

        X_shared, X_indep, X_text, Xf, Y, _ = self.extract_XY(df_wide, self.n_series, self.text_embed_dims)

        if not test:
            # Fit scalers
            sx_shared = StandardScaler().fit(X_shared) if X_shared is not None else None
            sx_indep = [StandardScaler().fit(X_indep[s]) for s in range(self.n_series)]

            # Text scalers: sx_text[s][e]
            sx_text = []
            for s in range(self.n_series):
                series_scalers = []
                for e in range(len(X_text[s])):
                    if X_text[s][e] is not None:
                        series_scalers.append(StandardScaler().fit(X_text[s][e]))
                    else:
                        series_scalers.append(None)
                sx_text.append(series_scalers)

            sxf = StandardScaler().fit(Xf) if Xf is not None else None
            sy = [StandardScaler().fit(Y[s]) for s in range(self.n_series)]

            # Transform
            Xs_shared = sx_shared.transform(X_shared).astype(np.float32) if X_shared is not None else None
            Xs_indep = [sx_indep[s].transform(X_indep[s]).astype(np.float32) for s in range(self.n_series)]

            # Transform text
            Xs_text = []
            for s in range(self.n_series):
                series_text = []
                for e in range(len(X_text[s])):
                    if X_text[s][e] is not None:
                        series_text.append(sx_text[s][e].transform(X_text[s][e]).astype(np.float32))
                    else:
                        series_text.append(None)
                Xs_text.append(series_text)

            Xs_f = sxf.transform(Xf).astype(np.float32) if Xf is not None else None
            Ys = [sy[s].transform(Y[s]).astype(np.float32) for s in range(self.n_series)]

            X_shared_seq, X_indep_seq, X_text_seq, Xf_seq, Y_seq = self.make_seq(
                Xs_shared, Xs_indep, Xs_text, Xs_f, Ys, T=32, stride=1
            )
            loader = DataLoader(
                self.XYSeqDataset(X_shared_seq, X_indep_seq, X_text_seq, Xf_seq, Y_seq),
                batch_size=64, shuffle=True, collate_fn=custom_collate_fn
            )

            fourier_cols = [c for c in df_wide.columns if c.startswith("fourier_")]
            F_total = len(fourier_cols)

            t0 = pd.to_datetime(df_wide["day"]).astype("int64") // 86_400_000_000_000

            return (loader, sx_shared, sx_indep, sx_text, sxf, sy, X_shared, X_indep, X_text, Xf, Y, F_total, t0.min())

        else:
            return X_shared, X_indep, X_text, Xf, Y

    @torch.no_grad()
    def forecast_knownX(self, X_shared_hist, X_indep_hist, X_text_hist, Xf_hist,
                        X_shared_fut, X_indep_fut, X_text_fut, Xf_fut, T,
                        sx_shared=None, sx_indep=None, sx_text=None, sxf=None, sy=None):
        self.eval()
        dev = next(self.parameters()).device

        # Process shared
        if X_shared_hist is not None:
            shared_hist = np.asarray(X_shared_hist[-T:], dtype=np.float32)
            shared_fut = np.asarray(X_shared_fut, dtype=np.float32)
            if sx_shared is not None:
                shared_hist = sx_shared.transform(shared_hist).astype(np.float32)
                shared_fut = sx_shared.transform(shared_fut).astype(np.float32)
            x_shared = torch.from_numpy(np.concatenate([shared_hist, shared_fut], axis=0)[None, ...]).to(dev)
        else:
            x_shared = None

        # Process indep per series
        x_indep = []
        for s in range(self.n_series):
            indep_hist = np.asarray(X_indep_hist[s][-T:], dtype=np.float32)
            indep_fut = np.asarray(X_indep_fut[s], dtype=np.float32)
            if sx_indep is not None:
                indep_hist = sx_indep[s].transform(indep_hist).astype(np.float32)
                indep_fut = sx_indep[s].transform(indep_fut).astype(np.float32)
            x_indep.append(torch.from_numpy(np.concatenate([indep_hist, indep_fut], axis=0)[None, ...]).to(dev))

        # Process text per series per embedding
        x_text = []
        for s in range(self.n_series):
            series_text = []
            for e in range(len(X_text_hist[s])):
                if X_text_hist[s][e] is not None:
                    text_hist = np.asarray(X_text_hist[s][e][-T:], dtype=np.float32)
                    text_fut = np.asarray(X_text_fut[s][e], dtype=np.float32)
                    if sx_text is not None and sx_text[s][e] is not None:
                        text_hist = sx_text[s][e].transform(text_hist).astype(np.float32)
                        text_fut = sx_text[s][e].transform(text_fut).astype(np.float32)
                    series_text.append(
                        torch.from_numpy(np.concatenate([text_hist, text_fut], axis=0)[None, ...]).to(dev))
                else:
                    series_text.append(None)
            x_text.append(series_text)

        # Process fourier
        if Xf_hist is not None:
            xf_hist = np.asarray(Xf_hist[-T:], dtype=np.float32)
            xf_fut = np.asarray(Xf_fut, dtype=np.float32)
            if sxf is not None:
                xf_hist = sxf.transform(xf_hist).astype(np.float32)
                xf_fut = sxf.transform(xf_fut).astype(np.float32)
            xf = torch.from_numpy(np.concatenate([xf_hist, xf_fut], axis=0)[None, ...]).to(dev)
        else:
            xf = None

        n_fut = len(X_indep_fut[0])
        o_all, _ = self(x_shared, x_indep, x_text, xf)

        # Output: [B, T, n_series, H]
        y_preds = []
        for s in range(self.n_series):
            y_std = o_all[0, -n_fut:, s, :].cpu().numpy()
            if sy is not None:
                y_preds.append(sy[s].inverse_transform(y_std))
            else:
                y_preds.append(y_std)

        return y_preds

def generate_prediction_intervals(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        epsilon_mean: np.ndarray,
        epsilon_cov: np.ndarray,
        confidence: float = 0.95,
        n_samples: int = 50000) -> Dict[str, np.ndarray]:

    y_pred = np.asarray(y_pred)
    H = len(epsilon_mean)  # Infer H from epsilon_mean shape

    epsilon_samples = np.random.multivariate_normal(
        np.zeros_like(epsilon_mean),
        epsilon_cov,
        size=n_samples
    )

    if y_pred.ndim == 1:
        y_samples = y_pred[np.newaxis, :] + epsilon_samples
    else:
        n_days = y_pred.shape[0]
        y_samples = np.zeros((n_samples, n_days, H))  # Use H
        for d in range(n_days):
            y_samples[:, d, :] = y_pred[d] + epsilon_samples

    alpha = 1 - confidence
    lower = np.quantile(y_samples, alpha / 2, axis=0)
    upper = np.quantile(y_samples, 1 - alpha / 2, axis=0)
    point = y_pred + epsilon_mean

    result = {
        'point': point,
        'lower': lower,
        'upper': upper,
        'width': upper - lower,
        'samples': y_samples
    }

    if y_true is not None:
        y_true = np.asarray(y_true)

        if y_pred.ndim == 1:
            p_values = np.zeros(H)  # Use H

            for h in range(H):  # Use H
                p_values[h] = np.mean(y_samples[:, h] <= y_true[h])

            result['p_values'] = p_values

        else:
            n_days = y_pred.shape[0]
            p_values = np.zeros((n_days, H))  # Use H

            for d in range(n_days):
                for h in range(H):  # Use H
                    p_values[d, h] = np.mean(y_samples[:, d, h] <= y_true[d, h])

            result['p_values'] = p_values

    return result


@dataclass
class training_config:
    n_epochs: int = 25
    device: torch.device = torch.device("cpu")
    T_hist: int = 32
    lr: float = 5e-4
    kl_coeff: float = 1.0
    pred_samples: int = 1000
    lambda0: float = 1e-5
    lambdaf: float = 5e-4


@dataclass
class fourier_config:
    mode: Literal["vector", "matrix"] = "vector"
    K_weekly: int = 0
    K_monthly: int = 0
    K_yearly: int = 0
    P_WEEK: float = 7.0
    P_MONTH: float = 365.25 / 12.0
    P_yearly: float = 365.25


class RNN_train_fourier:
    def __init__(self, model, training_config, fourier_conf, deterministic=True):
        self.model = model
        self.training_config = training_config
        self.fourier_conf = fourier_conf
        self.model.to(self.training_config.device)
        self.start_day = None
        self.deterministic = deterministic

    def __call__(self, df_train):
        loader, sx_shared, sx_indep, sx_text, sxf, sy, X_shared, X_indep, X_text, Xf, Y, F_total, start_day = self.model.get_dataloader(
            df_train, self.fourier_conf)
        self.start_day = start_day
        self.sx_shared = sx_shared
        self.sx_indep = sx_indep
        self.sx_text = sx_text
        self.sxf = sxf
        self.sy = sy
        self.X_shared = X_shared
        self.X_indep = X_indep
        self.X_text = X_text
        self.Xf = Xf
        self.Y = Y

        if self.model.mode == "vector":
            K_est = F_total // 2
        elif self.model.mode == "matrix":
            K_est = (F_total // (2 * self.model.H))
        else:
            raise ValueError("mode must be 'vector' or 'matrix'")

        harmonic_orders = sum(([k, k] for k in range(1, K_est + 1)), [])
        device = self.training_config.device
        opt = torch.optim.Adam(self.model.parameters(), lr=self.training_config.lr)

        # Training loop
        for ep in range(self.training_config.n_epochs):
            self.model.train()
            for x_shared_b, x_indep_b, x_text_b, xf_b, yb in loader:
                if x_shared_b is not None:
                    x_shared_b = x_shared_b.to(device)
                x_indep_b = [x.to(device) for x in x_indep_b]

                # Move text to device
                x_text_b_dev = []
                for s in range(self.model.n_series):
                    series_text = []
                    for e in range(len(x_text_b[s])):
                        if x_text_b[s][e] is not None:
                            series_text.append(x_text_b[s][e].to(device))
                        else:
                            series_text.append(None)
                    x_text_b_dev.append(series_text)

                if xf_b is not None:
                    xf_b = xf_b.to(device)
                yb = [y.to(device) for y in yb]

                opt.zero_grad()
                o, _ = self.model(x_shared_b, x_indep_b, x_text_b_dev, xf_b)

                # Loss: sum over all series
                data_loss = sum(F.mse_loss(o[:, :, s, :], yb[s]) for s in range(self.model.n_series))
                prior_loss = self.model.reg_loss(
                    lambda0=self.training_config.lambda0,
                    lambdaf=self.training_config.lambdaf,
                    harmonic_orders=harmonic_orders
                )
                loss = data_loss + prior_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                opt.step()
            print(f"epoch {ep + 1} loss: {float(loss):.4f}")

        self.model.eval()

        if not self.deterministic:
            T = self.training_config.T_hist
            n_series = self.model.n_series

            # Residuals per series
            train_residuals = [[] for _ in range(n_series)]

            with torch.no_grad():
                for i in range(T, len(self.Y[0])):
                    X_shared_hist = self.X_shared[:i] if self.X_shared is not None else None
                    X_indep_hist = [self.X_indep[s][:i] for s in range(n_series)]
                    Xf_hist = self.Xf[:i] if self.Xf is not None else None

                    X_shared_fut = self.X_shared[i:i + 1] if self.X_shared is not None else None
                    X_indep_fut = [self.X_indep[s][i:i + 1] for s in range(n_series)]
                    Xf_fut = self.Xf[i:i + 1] if self.Xf is not None else None

                    preds = self.model.forecast_knownX(
                        X_shared_hist, X_indep_hist,
                        [[self.X_text[s][e][:i] if self.X_text[s][e] is not None else None for e in
                          range(len(self.X_text[s]))] for s in range(n_series)],
                        Xf_hist,
                        X_shared_fut, X_indep_fut,
                        [[self.X_text[s][e][i:i + 1] if self.X_text[s][e] is not None else None for e in
                          range(len(self.X_text[s]))] for s in range(n_series)],
                        Xf_fut,
                        T=T, sx_shared=self.sx_shared, sx_indep=self.sx_indep, sx_text=self.sx_text, sxf=self.sxf,
                        sy=self.sy
                    )

                    for s in range(n_series):
                        y_true = self.Y[s][i]
                        residual = y_true - preds[s].flatten()
                        train_residuals[s].append(residual)

            self.epsilon_mean = [np.mean(train_residuals[s], axis=0) for s in range(n_series)]
            self.epsilon_cov = [np.cov(train_residuals[s], rowvar=False) for s in range(n_series)]
            self.epsilon_std = [np.sqrt(np.diag(self.epsilon_cov[s])) for s in range(n_series)]

    def forecast(self, df_test):
        X_shared_test, X_indep_test, X_text_test, Xf_test, Y_test = self.model.get_dataloader(
            df_test, self.fourier_conf, test=True, start_day=self.start_day
        )
        T = self.training_config.T_hist
        n_series = self.model.n_series

        with torch.no_grad():
            y_preds = self.model.forecast_knownX(
                self.X_shared, self.X_indep, self.X_text, self.Xf,
                X_shared_test, X_indep_test, X_text_test, Xf_test,
                T=T, sx_shared=self.sx_shared, sx_indep=self.sx_indep, sx_text=self.sx_text, sxf=self.sxf, sy=self.sy
            )

        if self.deterministic:
            return {
                'test_pred': [y_preds[s].flatten() for s in range(n_series)],
                'test_true': [Y_test[s].flatten() for s in range(n_series)],
            }

        results = {'test_pred': [], 'test_true': [], 'y_pred_lower': [], 'y_pred_upper': [], 'p_values': []}

        for s in range(n_series):
            intervals = generate_prediction_intervals(
                y_pred=y_preds[s].flatten(),
                y_true=Y_test[s].flatten(),
                epsilon_mean=self.epsilon_mean[s],
                epsilon_cov=self.epsilon_cov[s],
                confidence=0.95,
                n_samples=50000
            )
            results['test_pred'].append(y_preds[s].flatten())
            results['test_true'].append(Y_test[s].flatten())
            results['y_pred_lower'].append(intervals['lower'])
            results['y_pred_upper'].append(intervals['upper'])
            results['p_values'].append(intervals['p_values'].flatten())

        return results
