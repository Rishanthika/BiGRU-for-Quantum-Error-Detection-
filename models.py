"""models.py — Deep learning architectures for quantum error detection."""

import torch
import torch.nn as nn

PAD_IDX = 0
N_SIG   = 20   # number of hand-crafted signal features


class LSTMClassifier(nn.Module):
    """2-layer LSTM with mean + last-real-token pooling and signal feature fusion."""
    def __init__(self, vocab, emb_dim=256, hid=512, n_cls=4, layers=2, drop=0.25, bi=False, n_sig=N_SIG):
        super().__init__()
        self.emb  = nn.Embedding(vocab, emb_dim, padding_idx=PAD_IDX)
        nn.init.xavier_uniform_(self.emb.weight)
        self.emb.weight.data[PAD_IDX].fill_(0)
        self.lstm = nn.LSTM(emb_dim, hid, layers, batch_first=True,
                            dropout=drop if layers > 1 else 0, bidirectional=bi)
        D = 2 if bi else 1
        self.drop = nn.Dropout(drop); self.norm = nn.LayerNorm(hid * D)
        self.sig_proj = nn.Sequential(nn.Linear(n_sig, 64), nn.ReLU(), nn.LayerNorm(64), nn.Dropout(0.1))
        fused = hid * D * 2 + 64
        self.fc1 = nn.Linear(fused, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, n_cls)
        self.norm1 = nn.LayerNorm(512); self.norm2 = nn.LayerNorm(256)
        self.act = nn.GELU(); self.drop2 = nn.Dropout(drop)

    def forward(self, x, sig):
        emb = self.drop(self.emb(x))
        out, _ = self.lstm(emb); out = self.norm(out)
        pad_mask = (x != PAD_IDX); mask_f = pad_mask.unsqueeze(-1).float()
        mean_pool = (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        lengths = pad_mask.long().sum(1).clamp(min=1) - 1
        last_out = out[torch.arange(out.size(0)), lengths]
        sig_feat = self.sig_proj(sig)
        fused = self.drop(torch.cat([last_out, mean_pool, sig_feat], dim=1))
        h = self.act(self.norm1(self.fc1(fused))); h = self.drop2(h)
        h = self.act(self.norm2(self.fc2(h))); h = self.drop2(h)
        return self.fc3(h)


class GRUClassifier(nn.Module):
    """
    Quantum Error Detector using Gated Recurrent Units.

    GRU replaces LSTM's three-gate design (forget/input/output) with two gates
    (update + reset), yielding ~33% fewer parameters per layer. Bidirectional
    variant (bi=True) is especially effective at catching LOGICAL errors because
    the right-to-left pass reads future gate context before encountering a
    premature measure instruction.
    """
    def __init__(self, vocab, emb_dim=256, hid=512, n_cls=4, layers=2, drop=0.25, bi=False, n_sig=N_SIG):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD_IDX)
        nn.init.xavier_uniform_(self.emb.weight)
        self.emb.weight.data[PAD_IDX].fill_(0)
        self.gru = nn.GRU(emb_dim, hid, layers, batch_first=True,
                          dropout=drop if layers > 1 else 0, bidirectional=bi)
        D = 2 if bi else 1
        self.drop = nn.Dropout(drop); self.norm = nn.LayerNorm(hid * D)
        self.sig_proj = nn.Sequential(nn.Linear(n_sig, 64), nn.ReLU(), nn.LayerNorm(64), nn.Dropout(0.1))
        fused = hid * D * 2 + 64
        self.fc1 = nn.Linear(fused, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, n_cls)
        self.norm1 = nn.LayerNorm(512); self.norm2 = nn.LayerNorm(256)
        self.act = nn.GELU(); self.drop2 = nn.Dropout(drop)

    def forward(self, x, sig):
        emb = self.drop(self.emb(x))
        out, _ = self.gru(emb); out = self.norm(out)
        pad_mask = (x != PAD_IDX); mask_f = pad_mask.unsqueeze(-1).float()
        mean_pool = (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        lengths = pad_mask.long().sum(1).clamp(min=1) - 1
        last_out = out[torch.arange(out.size(0)), lengths]
        sig_feat = self.sig_proj(sig)
        fused = self.drop(torch.cat([last_out, mean_pool, sig_feat], dim=1))
        h = self.act(self.norm1(self.fc1(fused))); h = self.drop2(h)
        h = self.act(self.norm2(self.fc2(h))); h = self.drop2(h)
        return self.fc3(h)


class TransformerClassifier(nn.Module):
    """
    Pre-LN Transformer encoder with CLS token and signal feature fusion.
    8 attention heads, 8 layers, ff_dim=1280, emb_dim=320.
    """
    def __init__(self, vocab, emb_dim=320, n_heads=8, n_layers=8, n_cls=4,
                 maxlen=192, drop=0.10, ff_dim=1280, n_sig=N_SIG):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD_IDX)
        nn.init.xavier_uniform_(self.tok_emb.weight)
        self.tok_emb.weight.data[PAD_IDX].fill_(0)
        self.pos_emb = nn.Embedding(maxlen + 2, emb_dim)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_tok, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=drop, batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.drop = nn.Dropout(drop)
        self.sig_proj = nn.Sequential(nn.Linear(n_sig, 64), nn.GELU(), nn.LayerNorm(64), nn.Dropout(0.1))
        fused = emb_dim + 64
        self.head = nn.Sequential(
            nn.LayerNorm(fused), nn.Linear(fused, 512), nn.GELU(), nn.Dropout(drop),
            nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, n_cls))

    def forward(self, x, sig):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        emb = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        cls = self.cls_tok.expand(B, -1, -1)
        seq = torch.cat([cls, emb], dim=1)
        pad_m = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), x == PAD_IDX], dim=1)
        out = self.encoder(seq, src_key_padding_mask=pad_m)
        sig_out = self.sig_proj(sig)
        return self.head(torch.cat([out[:, 0], sig_out], dim=1))
