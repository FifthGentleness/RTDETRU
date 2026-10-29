# RGCSA-V2: Bounded Geometric Relation Guidance + Geometry Message Passing (content path).
#
# 基于 v1 (decoder_rgc_sa.py) 的诊断结论重构:
#   诊断1: v1 偏置无界, layer1 偏置 std~141 >> 内容 logits O(1~5), softmax 饱和
#          -> 内容通路(QK)梯度被饿死, 注意力退化为几何硬路由 (clamp 实验证实模型
#             实际只使用 +-4~8 的动态范围)
#   诊断2: logit 偏置只能改注意力混合权重, 内容永远流不进 bbox_head/score_head
#          -> val_giou 与 base 持平, AP75 段零贡献
#
# V2 = 通路A(关系打分修复) + 通路B(几何消息传递, 补充内容通道):
#
#   通路A: rel(5维) -> 共享RelEncoder -> 每层LayerScale(零初始化)
#          -> bias = bias_max * tanh(.)  有界注入 attn_mask
#          rel = [tanh(dxy/6)(2), clamp(log_ratio,+-4)(2), IoU_ij(1,detach)]
#   通路B: w_ij = norm(ReLU(tanh(logits_ij.detach())))   纯几何路由(detach, 不抢梯度)
#          nbr_i = sum_j w_ij * embed_j                  bmm 一次
#          embed += gate(零初始化) * MLP(nbr)            B-heavy: hidden=256, 双层
#
# step-0 等价: ls 与 gate 均零初始化 -> 训练起点与 base 逐位相同.
# DN 兼容: DN query 仍走原始掩码自注意力 (_InnovSelfAttnLayer 保证), V2 仅作用于
#          matching-matching 块.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_innov_common import (_InnovSelfAttnLayer, _RTDETRDecoderInnov,
                                   DeformableTransformerDecoderInnov,
                                   _pairwise_iou)

__all__ = ('DecoderLayerRGCSAV2', 'RTDETRDecoderRGCSAV2')


class RelEncoder(nn.Module):
    """跨层共享的相对几何编码器: rel(5维) -> 每头 logit."""

    def __init__(self, n_heads=8, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(5, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_heads))

    def forward(self, rel):
        return self.mlp(rel)


class DecoderLayerRGCSAV2(_InnovSelfAttnLayer):
    """RGCSA-V2 layer: 有界几何偏置 + 几何消息传递残差."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.,
                 act=nn.ReLU(), n_levels=4, n_points=4, rel_hidden=64,
                 gmp_hidden=256, gmp_layers=2, bias_max=4.0):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points)
        self.n_heads = n_heads
        self.d_model = d_model
        self.bias_max = float(bias_max)
        # 共享 RelEncoder 由 head 类在构建后注入(所有层同一实例)
        self.rel_enc = None
        # 每层关系 logit 的 LayerScale, 零初始化 = step-0 等价 + 防失控头
        self.ls = nn.Parameter(torch.zeros(n_heads))
        # 通路B: GMPr (B-heavy: hidden=256 双层 MLP)
        dims = [d_model] + [gmp_hidden] * gmp_layers + [d_model]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.GELU())
        self.gmp = nn.Sequential(*mods)
        self.gate = nn.Parameter(torch.zeros(d_model))

    def _rel_feat(self, boxes):
        """boxes [B,N,4] cxcywh[0,1] -> rel [B,N,N,5]."""
        xy, wh = boxes[..., :2], boxes[..., 2:].clamp(min=1e-4)
        scale = (wh[:, :, None, :] * wh[:, None, :, :]).clamp(min=1e-8).sqrt()
        dxy = torch.tanh((xy[:, :, None, :] - xy[:, None, :, :]) / (6.0 * scale))
        log_ratio = torch.log((wh[:, :, None, :] / wh[:, None, :, :]).clamp(min=1e-6))
        log_ratio = log_ratio.clamp(-4.0, 4.0)
        iou = _pairwise_iou(boxes).detach().unsqueeze(-1)
        return torch.cat([dxy, log_ratio, iou], dim=-1)

    def _innov_attn(self, mt, ref_xy, pos, boxes):
        b, n = mt.shape[:2]
        q = k = mt if pos is None else mt + pos

        # ---- 通路A: 有界几何注意力偏置 ----
        rel = self._rel_feat(boxes)                       # [B,N,N,5]
        logits = self.rel_enc(rel) * self.ls[None, None, None, :]   # [B,N,N,nh]
        bias = self.bias_max * torch.tanh(logits)
        bias = bias.permute(0, 3, 1, 2).reshape(b * self.n_heads, n, n)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                             mt.transpose(0, 1), attn_mask=bias)[0].transpose(0, 1)
        mt2 = self.norm1(mt + self.dropout1(tgt))

        # ---- 通路B: 几何消息传递(补充内容通道) ----
        w = F.relu(torch.tanh(logits.detach())).mean(dim=-1)        # [B,N,N]
        w = w / (w.sum(-1, keepdim=True) + 1e-6)
        nbr = torch.bmm(w, mt2)                                     # [B,N,d]
        mt2 = mt2 + self.gate * self.gmp(nbr)
        return mt2


class RTDETRDecoderRGCSAV2(_RTDETRDecoderInnov):
    """RTDETRDecoder + RGCSA-V2.

    Args (除 RTDETRDecoder 标准参数外, 均可用默认值):
        rel_hidden (int): 共享 RelEncoder 隐层宽度 (默认 64)
        gmp_hidden (int): GMPr MLP 隐层宽度 (默认 256, B-heavy)
        gmp_layers (int): GMPr MLP 隐层层数 (默认 2, B-heavy)
        bias_max (float): 通路A 偏置幅值上界 (默认 4.0, clamp 实验依据)
    """

    def _rebuild_decoder(self, hd, nh, ndl, d_ffn, dropout, act, ndp, eval_idx,
                         **kw):
        layer = DecoderLayerRGCSAV2(hd, nh, d_ffn, dropout, act, self.nl, ndp,
                                    rel_hidden=kw.get('rel_hidden', 64),
                                    gmp_hidden=kw.get('gmp_hidden', 256),
                                    gmp_layers=kw.get('gmp_layers', 2),
                                    bias_max=kw.get('bias_max', 4.0))
        dec = DeformableTransformerDecoderInnov(hd, layer, ndl, eval_idx)
        # 跨层共享 RelEncoder(同一实例, 参数只存一份, 梯度跨层累计)
        shared = RelEncoder(nh, kw.get('rel_hidden', 64))
        for l in dec.layers:
            l.rel_enc = shared
        return dec

    def extra_repr(self):
        ls = [float(l.ls.detach().abs().mean()) for l in self.decoder.layers]
        gt = [float(l.gate.detach().abs().mean()) for l in self.decoder.layers]
        return (f'RGCSA-V2 | bias_max={self.decoder.layers[0].bias_max} '
                f'| ls_mean={[round(x, 5) for x in ls]} '
                f'| gate_mean={[round(x, 5) for x in gt]}')
