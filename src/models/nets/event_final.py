from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from src.models.utils.embedding import PositionEmbedding, timestep_embedding

class MiniConformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        kernel_size: int = 31,
    ):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.modebug_attention_observation = False
        self.modebug_attention_name = None
        self.modebug_attention_records = None
        self.modebug_attention_context = {}
        self.conv_ln = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout_c = nn.Dropout(dropout)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def set_modebug_attention_observation(
        self,
        enabled: bool,
        records: Optional[List[Dict[str, Any]]] = None,
        name: Optional[str] = None,
    ):
        self.modebug_attention_observation = enabled
        self.modebug_attention_records = records
        self.modebug_attention_name = name

    def set_modebug_attention_context(self, context: Optional[Dict[str, Any]] = None):
        self.modebug_attention_context = dict(context or {})

    def _record_modebug_cross_attention(self, weights: torch.Tensor):
        if self.modebug_attention_records is None:
            return
        record = dict(self.modebug_attention_context)
        record.update(
            {
                "module": self.modebug_attention_name,
                "kind": "cross_attn",
                "shape": list(weights.shape),
                "weights": weights.detach().cpu(),
            }
        )
        self.modebug_attention_records.append(record)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        cross: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ):
        x = x + 0.5 * self.ffn1(x)

        out, attn = self.mha(
            x, x, x,
            key_padding_mask=pad_mask,
            need_weights=False
        )
        x = x + out

        if cross is not None:
            key_padding = None
            if cross_mask is not None:
                key_padding = ~cross_mask
            if self.modebug_attention_observation:
                cross_out, cross_weights = self.cross_attn(
                    x, cross, cross,
                    key_padding_mask=key_padding,
                    need_weights=True,
                    average_attn_weights=False,
                )
                self._record_modebug_cross_attention(cross_weights)
                x = x + cross_out
            else:
                x = x + self.cross_attn(
                    x, cross, cross,
                    key_padding_mask=key_padding,
                    need_weights=False
                )[0]

        y = self.conv_ln(x)
        y = y.transpose(1, 2)
        y = self.pw_conv1(y)
        y = self.glu(y)
        y = self.dw_conv(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.pw_conv2(y)
        y = self.dropout_c(y.transpose(1, 2))
        x = x + y

        x = x + 0.5 * self.ffn2(x)
        return x

class LocalModule(nn.Module):
    def __init__(self, model_dim: int, num_groups: int = 16, mask_padding: bool = True):
        super().__init__()
        self.mask_padding = mask_padding
        self.conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1),
            nn.Conv1d(model_dim, model_dim, 3, padding=1, groups=model_dim),
            nn.GroupNorm(num_groups, model_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x, x_mask, y, y_mask, z=None):
        if self.mask_padding:
            x = x.masked_fill(~x_mask.unsqueeze(-1), 0)
        x = self.norm(x + self.conv(x.transpose(1, 2)).transpose(1, 2))
        return x, y

class MixedModule(nn.Module):
    def __init__(
        self,
        model_dim: int,
        patch_size: int = 8,
        dropout: float = 0.1,
        mask_padding: bool = True,
        conformer_cfg: dict | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_padding = mask_padding
        self.local_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 1),
            nn.ReLU(),
            nn.Conv1d(
                model_dim, model_dim, patch_size, stride=patch_size, groups=model_dim
            ),
        )
        cfg = dict(nhead=8, dim_ff=1024, dropout=dropout, kernel_size=31)
        if conformer_cfg:
            cfg.update(conformer_cfg)
        self.global_conformer = MiniConformer(model_dim, **cfg)

        self.f_func = nn.Linear(model_dim * 2, model_dim)
        self.fuse_fn = nn.Linear(model_dim * 2, model_dim)

        self.final_fc = nn.Linear(model_dim * 2, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def inject_text(self, seg, y):
        y_rep = y.repeat_interleave(seg.size(1), dim=1)
        gate = torch.sigmoid(self.f_func(torch.cat([seg, y_rep], dim=-1)))
        return self.fuse_fn(torch.cat([seg, y_rep * gate], dim=-1))

    def forward(
        self,
        x, x_mask,
        y, y_mask,
        d: Optional[torch.Tensor] = None,
        d_mask: Optional[torch.Tensor] = None,
    ):
        if self.mask_padding:
            x = x.masked_fill(~x_mask.unsqueeze(-1), 0)

        seq = x[:, 1:]
        B, L, D = seq.shape
        pad = (self.patch_size - L % self.patch_size) % self.patch_size
        if pad:
            seq = torch.cat([seq, seq.new_zeros(B, pad, D)], 1)

        seg = self.local_conv(seq.transpose(1, 2)).transpose(1, 2)
        seg = self.inject_text(seg, y)

        seg = self.global_conformer(
            seg,
            pad_mask=None,
            cross=d,
            cross_mask=d_mask,
        )

        seg_up = repeat(seg, "b l d -> b (l s) d", s=self.patch_size)
        fused = self.final_fc(torch.cat([seq, seg_up], -1))
        if pad:
            fused = fused[:, :-pad]

        out = torch.cat([x[:, :1], fused], 1)
        out = self.norm(out)
        return out, y

class StageBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        dim,
        mask_padding=True,
        num_groups=16,
        patch_size=8,
        dropout=0.1,
        conformer_cfg=None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()
        self.y_proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()

        self.local1 = LocalModule(dim, num_groups, mask_padding)
        self.mixed = MixedModule(
            dim,
            patch_size=patch_size,
            dropout=dropout,
            mask_padding=mask_padding,
            conformer_cfg=conformer_cfg,
        )
        self.local2 = LocalModule(dim, num_groups, mask_padding)

    def forward(
        self,
        x, x_mask,
        y, y_mask,
        d: Optional[torch.Tensor] = None,
        d_mask: Optional[torch.Tensor] = None,
    ):
        x, y = self.input_proj(x), self.y_proj(y)
        if d is not None:
            d = self.y_proj(d)

        x, _ = self.local1(x, x_mask, y, y_mask)
        x, _ = self.mixed(x, x_mask, y, y_mask, d, d_mask)
        x, _ = self.local2(x, x_mask, y, y_mask)
        return x, y

class EventT2M(nn.Module):
    def __init__(
        self,
        motion_dim: int = 263,
        max_motion_len: int = 196,
        text_dim: int = 512,
        pos_emb: str = "cos",
        dropout: float = 0.1,
        stage_dim: str = "256",
        num_groups: int = 16,
        patch_size: int = 8,
        conformer_cfg: dict | None = None,
        ssm_cfg=None,
        rms_norm=None,
        fused_add_norm=None,
        **_ignored,
    ):
        super().__init__()
        if "*" in stage_dim:
            b = int(stage_dim.split("*")[0])
            stage_dims = [b] * int(stage_dim.split("*")[1])
        else:
            stage_dims = [int(x) for x in stage_dim.split("-")]
        base = stage_dims[0]
        
        if pos_emb == "cos":
            self.pos_emb = PositionEmbedding(max_motion_len, base, dropout=0.1)
        elif pos_emb == "learn":
            self.pos_emb = PositionEmbedding(
                max_motion_len, base, dropout=0.1, grad=True
            )
        else:
            raise ValueError(f"pos_emb {pos_emb} not supported")
        self.m_in = nn.Linear(motion_dim, base)
        self.t_in = nn.Linear(text_dim, base)
        self.time_emb = nn.Linear(base, base)
        
        modules: List[nn.Module] = []
        in_dim = base
        for dim in stage_dims:
            modules.append(
                StageBlock(
                    in_dim,
                    dim,
                    num_groups=num_groups,
                    patch_size=patch_size,
                    dropout=dropout,
                    conformer_cfg=conformer_cfg,
                )
            )
            in_dim = dim
        self.layers = nn.ModuleList(modules)
        
        self.m_out = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_dim, motion_dim),
        )
        self.modebug_attention_records: List[Dict[str, Any]] = []

    def set_modebug_attention_observation(self, enabled: bool, clear: bool = True):
        if clear:
            self.clear_modebug_attention_observation()
        for idx, blk in enumerate(self.layers):
            blk.mixed.global_conformer.set_modebug_attention_observation(
                enabled,
                self.modebug_attention_records,
                f"denoiser.layers.{idx}.mixed.global_conformer.cross_attn",
            )

    def clear_modebug_attention_observation(self):
        self.modebug_attention_records.clear()

    def set_modebug_attention_context(self, **context):
        for blk in self.layers:
            blk.mixed.global_conformer.set_modebug_attention_context(context)

    def get_modebug_attention_records(self):
        return self.modebug_attention_records

    def forward(
        self,
        motion: torch.Tensor,
        motion_mask: torch.Tensor,
        timestep: torch.Tensor,
        text: Dict[str, torch.Tensor],
        decomposed_embed: Dict[str, torch.Tensor],
        decomposed_mask: torch.Tensor,
    ):
        x = self.m_in(motion)
        time_tok = self.time_emb(
            timestep_embedding(timestep, x.size(-1))
        ).unsqueeze(1)

        text_emb = text["text_emb"]
        text_tok = self.t_in(text_emb).unsqueeze(1)

        decomposed_tok = self.t_in(decomposed_embed["text_emb"])
        d_mask = decomposed_mask

        x = torch.cat([time_tok, x], 1)
        x_mask = torch.cat(
            [torch.ones_like(time_tok[..., 0], dtype=torch.bool),
             motion_mask], 1
        )
        x = self.pos_emb(x)

        for blk in self.layers:
            x, text_tok = blk(
                x, x_mask,
                text_tok, torch.ones_like(text_tok[..., 0], dtype=torch.bool),
                decomposed_tok, d_mask
            )

        return self.m_out(x[:, 1:])
