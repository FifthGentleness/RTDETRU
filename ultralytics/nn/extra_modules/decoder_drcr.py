"""
DRCR: Delta-Rule Cross-Reading (RT-DETR decoder innovation).

Inspired by:
  - Gated Delta Networks (ICLR 2025): linear attention with an exact
    delta-rule state update, where new evidence REPLACES similar stored
    content instead of accumulating on top of it, plus a data-dependent
    forgetting gate.

Mechanism:
  - Multi-scale feature tokens are scanned in Morton (Z-order) order and a
    per-head matrix state S is carried chunk-by-chunk with the EXACT
    closed-form delta-rule chunk update:
        A   = tril(diag(beta) K K^T, -1)
        w   = (I + A)^{-1} (beta*K)            ->  P = I - w^T K     (key side)
        Vc  = V - triu(V K^T, 1) @ W_flipped   (value side, suffix-corrected)
        S   = gamma * (S P) + (beta*K)^T Vc
    so redundant background tokens OVERWRITE each other (implicit evidence
    de-duplication in state space) while isolated small-object evidence
    survives, and the data-dependent gate gamma controls how much past
    evidence each chunk retains.
  - Keys are L2-normalized per token, which makes the write strength beta the
    EXACT replacement rate of the delta rule (stability requires
    beta * ||k||^2 <= 1, guaranteed by unit-norm keys and beta = sigmoid < 1).
  - The scan runs once per image in float64 (forward + backward directions);
    EVERY decoder layer reads the same evidence states with its own readout
    head, so each stage of box refinement re-queries the de-duplicated
    evidence as the reference boxes improve.
  - Every decoder query reads the delta-rule state of the Morton chunk
    containing its reference point from both directions (the union covers
    the full spatial extent), with linear-attention normalization by a
    denominator state.

Cross-attention of matching queries is replaced; dn queries keep the original
MSDeformAttn path and self-attention is the baseline masked attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer
from ultralytics.nn.modules.utils import inverse_sigmoid

from .decoder_innov_common import (_InnovCrossAttnLayer, _RTDETRDecoderInnov,
                                   _morton_codes, _token_centers,
                                   DeformableTransformerDecoderInnov)

__all__ = ('DeltaRuleReader', 'DRCRReadout', 'DecoderLayerDRCR',
           'DeformableTransformerDecoderDRCR', 'RTDETRDecoderDRCR')


def _unit(x, eps=1e-6):
    """L2-normalize the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def _tri_solve(A, B):
    """Solve (I + A) X = B for strictly-lower-triangular A.
    The solve runs in float64 (conditioning-sensitive) and returns float32;
    all other scan math stays in float32 for GPU throughput."""
    I = torch.eye(A.shape[-1], dtype=torch.float64, device=A.device)
    X = torch.linalg.solve_triangular(I + A.double(), B.double(), upper=False, left=True)
    return X.float()


class DeltaRuleReader(nn.Module):
    """Shared bidirectional chunk-recurrent delta-rule evidence builder.

    Built ONCE per image (the encoder features do not change across decoder
    layers) and read by every decoder layer through its own DRCRReadout head.
    """

    def __init__(self, d_model=256, n_heads=4, grid=64, chunk_len=64):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.grid = grid
        self.chunk_len = chunk_len

        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, n_heads)             # write strength (sigmoid)
        self.gamma_proj = nn.Linear(d_model, n_heads)            # per-head forget gate (sigmoid)
        self.norm = nn.LayerNorm(d_model)

    def build_states(self, feats, shapes, padding_mask=None):
        """Scan the multi-scale features once and return the evidence states.

        Args:
            feats (Tensor): [B, T, C] flattened multi-scale features.
            shapes (List):  [(h, w), ...] per-level shapes.

        Returns:
            dict with chunk-boundary states S_f/S_b [B,H,nc,D,D], denominator
            states n_f/n_b [B,H,nc,D], sorted morton codes and chunk length.
        """
        B, T, C = feats.shape
        H, D = self.n_heads, self.dk
        centers = _token_centers(shapes, feats.device, feats.dtype)
        x = self.norm(feats)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask[..., None], 0.0)

        codes = _morton_codes(centers.unsqueeze(0).expand(B, -1, -1), self.grid)
        order = codes.argsort(1)
        xs = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, C))
        sorted_codes = torch.gather(codes, 1, order)

        k = _unit(self.k_proj(xs)).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D] unit keys
        v = self.v_proj(xs).view(B, T, H, D).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(xs)).view(B, T, H, 1).transpose(1, 2)
        gamma = torch.sigmoid(self.gamma_proj(xs)).view(B, T, H, 1).transpose(1, 2)

        # forward and backward scans batched (backward = scan of the reversed order)
        S2, n2 = self._scan(torch.cat([k, torch.flip(k, [2])], 0),
                            torch.cat([v, torch.flip(v, [2])], 0),
                            torch.cat([beta, torch.flip(beta, [2])], 0),
                            torch.cat([gamma, torch.flip(gamma, [2])], 0))
        S_f, S_b = S2[:B], torch.flip(S2[B:], [2])
        n_f, n_b = n2[:B], torch.flip(n2[B:], [2])                    # realign chunk ids
        return dict(S_f=S_f, n_f=n_f, S_b=S_b, n_b=n_b,
                    sorted_codes=sorted_codes, chunk_len=self.chunk_len)

    def _chunk_terms(self, K, V, Bt):
        """Batched per-chunk delta-rule terms (independent across chunks).

        Args:
            K (Tensor): [B, H, nc, L, D] unit-norm keys (float64).
            V (Tensor): [B, H, nc, L, D] values (float64).
            Bt (Tensor):[B, H, nc, L, 1] write strengths (float64).

        Returns:
            (M, W, BK): per-chunk P-factors M [B,H,nc,D,D] (P = I - M),
            write terms W [B,H,nc,D,D], and beta-scaled keys BK [B,H,nc,L,D].
        """
        BK = Bt * K
        A = torch.tril(Bt * (K @ K.transpose(-1, -2)), -1)
        w = _tri_solve(A, BK)
        M = w.transpose(-1, -2) @ K                        # sum_t w_t k_t^T
        # value side: suffix-corrected values via the reversed WY transform
        Kf, Vf, Bf = torch.flip(K, [3]), torch.flip(V, [3]), torch.flip(Bt, [3])
        BKf = Bf * Kf
        Af = torch.tril(Bf * (Kf @ Kf.transpose(-1, -2)), -1)
        wf = _tri_solve(Af, BKf)
        Wflip = torch.flip(wf, [3])
        Vcorr = V - torch.triu(V @ K.transpose(-1, -2), 1) @ Wflip
        W = BK.transpose(-1, -2) @ Vcorr                   # [B,H,nc,D,D]
        return M, W, BK

    def _scan(self, k, v, beta, gamma):
        """Exact delta-rule state carry over chunks (one direction).

        Args:
            k (Tensor):    [B, H, T, D] unit-norm keys.
            v (Tensor):    [B, H, T, D] values.
            beta (Tensor): [B, H, T, 1] write strength in (0, 1).
            gamma (Tensor):[B, H, T, D] per-dim forgetting gate in (0, 1).

        Returns:
            (S, n): chunk-boundary states [B, H, nc, D, D] (retrieval y = S^T k)
            and denominator states [B, H, nc, D]; state i covers chunks 0..i.
        """
        B, H, T, D = k.shape
        chunk = min(self.chunk_len, T)
        nc = (T + chunk - 1) // chunk
        pad = nc * chunk - T
        if pad:
            k = torch.cat([k, k[:, :, -1:].expand(-1, -1, pad, -1)], 2)
            v = torch.cat([v, v[:, :, -1:].expand(-1, -1, pad, -1)], 2)
            beta = torch.cat([beta, beta[:, :, -1:].expand(-1, -1, pad, 1)], 2)
            gamma = torch.cat([gamma, gamma[:, :, -1:].expand(-1, -1, pad, -1)], 2)

        K = k.view(B, H, nc, chunk, D)
        V = v.view(B, H, nc, chunk, D)
        Bt = beta.view(B, H, nc, chunk, 1)
        G = gamma.view(B, H, nc, chunk, 1)
        g_chunk = G.clamp(min=1e-4).log().mean(3).exp()    # geometric-mean retention [B,H,nc,1]

        M_all, W_all, BK_all = self._chunk_terms(K, V, Bt)
        BK_sum = BK_all.sum(3)                             # [B,H,nc,D]

        S = torch.zeros(B, H, D, D, dtype=k.dtype, device=k.device)
        n = torch.zeros(B, H, D, dtype=k.dtype, device=k.device)
        S_list, n_list = [], []
        for c in range(nc):
            Mc, Wc, gc = M_all[:, :, c], W_all[:, :, c], g_chunk[:, :, c]
            S = gc.unsqueeze(-1) * (S - S @ Mc) + Wc
            n = gc * (n.unsqueeze(-2) @ Mc).squeeze(-2) + BK_sum[:, :, c]
            S_list.append(S)
            n_list.append(n)
        S = torch.stack(S_list, 2)
        n = torch.stack(n_list, 2)
        return S, n                                        # [B,H,nc,D,D], [B,H,nc,D]


class DRCRReadout(nn.Module):
    """Per-layer readout head over the shared delta-rule evidence states."""

    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.SiLU())
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_q = nn.LayerNorm(d_model)

    def forward(self, q, ref_boxes, states):
        """Cross-read for matching queries.

        Args:
            q (Tensor):    [B, N, C] query content + positional embedding.
            ref_boxes:     [B, N, 4] cxcywh reference boxes in [0, 1].
            states (dict): shared evidence states from DeltaRuleReader.

        Returns:
            Tensor: [B, N, C] cross-attention output (pre-residual).
        """
        B, N, C = q.shape
        H, D = self.n_heads, self.dk
        T = states['sorted_codes'].shape[1]
        nc = states['S_f'].shape[2]

        chunk = states['chunk_len']
        q_codes = _morton_codes(ref_boxes[..., :2], 64)
        pos = torch.searchsorted(states['sorted_codes'], q_codes).clamp(0, T - 1)
        cid = (pos // chunk).clamp(0, nc - 1)                             # [B, N]

        kq = _unit(self.q_proj(self.norm_q(q))).view(B, N, H, D)          # [B,N,H,D]
        Sfg = torch.gather(states['S_f'], 2, cid.view(B, 1, -1, 1, 1).expand(-1, H, -1, D, D)).permute(0, 2, 1, 3, 4)
        nfg = torch.gather(states['n_f'], 2, cid.view(B, 1, -1, 1).expand(-1, H, -1, D)).permute(0, 2, 1, 3)
        Sbg = torch.gather(states['S_b'], 2, cid.view(B, 1, -1, 1, 1).expand(-1, H, -1, D, D)).permute(0, 2, 1, 3, 4)
        nbg = torch.gather(states['n_b'], 2, cid.view(B, 1, -1, 1).expand(-1, H, -1, D)).permute(0, 2, 1, 3)

        def _read(S, n):
            y = torch.einsum('bnhd,bnhde->bnhe', kq, S)                   # [B,N,H,D] (= S^T k)
            den = torch.einsum('bnhd,bnhd->bnh', kq, n).unsqueeze(-1).clamp(min=1e-4)
            return y / den

        y = self.fuse(torch.cat([_read(Sfg, nfg).reshape(B, N, C),
                                 _read(Sbg, nbg).reshape(B, N, C)], -1))
        return self.out_proj(y)


class DecoderLayerDRCR(_InnovCrossAttnLayer):
    """Baseline decoder layer whose matching-query cross-attention is replaced
    by a readout over the shared DRCR delta-rule evidence states."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(),
                 n_levels=4, n_points=4, scan_heads=4):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.readout = DRCRReadout(d_model, scan_heads)

    def _innov_cross(self, q, embed, ref_boxes, feats, shapes, padding_mask=None,
                     shared_states=None):
        assert shared_states is not None, 'DRCR layer requires shared evidence states'
        return self.readout(q, ref_boxes, shared_states)


class DeformableTransformerDecoderDRCR(DeformableTransformerDecoderInnov):
    """Decoder wrapper that scans the features ONCE and feeds the shared
    delta-rule evidence states to every decoder layer."""

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, reader=None):
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.reader = reader

    def forward(self, embed, refer_bbox, feats, shapes, bbox_head, score_head,
                pos_mlp, attn_mask=None, padding_mask=None, dn_num=0):
        states = self.reader.build_states(feats, shapes, padding_mask)
        output = embed
        dec_bboxes, dec_cls = [], []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            extra = dict(dn_num=dn_num, dec_boxes=refer_bbox, shared_states=states) \
                if getattr(layer, 'INNOV_KWARGS', False) else {}
            output = layer(output, refer_bbox, feats, shapes, padding_mask,
                           attn_mask, pos_mlp(refer_bbox), **extra)

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            if i == self.num_layers - 1:
                break
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class RTDETRDecoderDRCR(_RTDETRDecoderInnov):
    """RT-DETR decoder + DRCR (delta-rule cross-reading, shared states).

    Extra yaml args (after the standard RTDETRDecoder args):
        scan_heads (int, default 4): heads of the delta-rule state.
        chunk_len (int, default 64): Morton chunk length for the exact update.
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         scan_heads=4, chunk_len=64, **_):
        layer = DecoderLayerDRCR(hd, nh, d_ffn, dropout, act, self.nl, ndp, scan_heads)
        reader = DeltaRuleReader(hd, scan_heads, chunk_len=chunk_len)
        return DeformableTransformerDecoderDRCR(hd, layer, ndl, eval_idx, reader=reader)
