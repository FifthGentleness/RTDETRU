# Ultralytics YOLO 棣冩偘, AGPL-3.0 license
"""
Loss plugin registry for RT-DETR loss ablations.

How to add a new loss (tasks.py NEVER needs to change again):
    1. Create a self-contained plugin file in this package, e.g.
       `ultralytics/utils/loss_plugins/your_loss.py`, containing a subclass of
       `RTDETRDetectionLoss` (or `RTDETRDetectionLossSD`-style subclass) that
       overrides ONLY the loss component it changes.
    2. Register it below:
           LOSS_REGISTRY['your_loss'] = 'ultralytics.utils.loss_plugins.your_loss.YourLossClass'
    3. Enable it in the model yaml:
           loss_name: your_loss
           loss_params:            # optional, forwarded as constructor kwargs
               your_alpha: 0.7

Rules of thumb:
    - Plugins that replace the SAME loss term (e.g. the GIoU term) are mutually
      exclusive 閳?`loss_name` selects exactly one.
    - Keep each plugin focused on a single override point (`_get_loss_bbox`,
      `_get_loss_class`, matcher, ...). For orthogonal losses that must be
      stacked, add a dedicated combined plugin class instead of multi-switching
      one class.
"""

import importlib

# name -> full dotted path of the loss class
LOSS_REGISTRY = {
    'sd': 'ultralytics.utils.loss_plugins.sd_loss.RTDETRDetectionLossSD',
    'tri_scale': 'ultralytics.utils.loss_plugins.tri_scale_loss.TriScaleDetectionLoss',
    'sual': 'ultralytics.utils.loss_plugins.sual_loss.RTDETRDetectionLossSUAL',
    'sual_ur': 'ultralytics.utils.loss_plugins.sual_ur_loss.RTDETRDetectionLossSUALUR',
    'shape_iou': 'ultralytics.utils.loss_plugins.shape_iou_loss.RTDETRDetectionLossShapeIoU',
    'mal_shape': 'ultralytics.utils.loss_plugins.mal_shape_loss.RTDETRDetectionLossMalShape',
    'inner_siou': 'ultralytics.utils.loss_plugins.inner_siou_loss.RTDETRDetectionLossInnerSIoU',
    'inner_mpdiou': 'ultralytics.utils.loss_plugins.inner_mpdiou_loss.RTDETRDetectionLossInnerMPDIoU',
    # InterpIoU loss family (Liu & Watanabe, Neurocomputing 2026):
    #   inner_interpiou    -> D-InterpIoU + Inner-IoU auxiliary box
    #   focaler_interpiou  -> D-InterpIoU + Focaler interval-mapping auxiliary term
    #   da_inner_interpiou -> D-InterpIoU + Dynamic-Adaptive Inner auxiliary box
    # Pure InterpIoU / D-InterpIoU (no Inner-IoU auxiliary term), official
    # formulation: InterpIoU = IoU + IoU(B_interp, B_gt) - 1.
    'interpiou': 'ultralytics.utils.loss_plugins.interpiou_loss.RTDETRDetectionLossInterpIoU',
    'inner_interpiou': 'ultralytics.utils.loss_plugins.inner_interpiou_loss.RTDETRDetectionLossInnerInterpIoU',
    'focaler_interpiou': 'ultralytics.utils.loss_plugins.focaler_interpiou_loss.RTDETRDetectionLossFocalerInterpIoU',
    'da_inner_interpiou': 'ultralytics.utils.loss_plugins.da_inner_interpiou_loss.RTDETRDetectionLossDAInnerInterpIoU',
    # Mal-InterpIoU: DEIM MAL (CVPR 2025) classification + Dynamic-InterpIoU
    # regression, with optional interpolation-refined matchability coupling.
    # NOTE: unlike the pure-regression variants above, this changes the
    # classification loss too.
    'mal_interpiou': 'ultralytics.utils.loss_plugins.mal_interpiou_loss.RTDETRDetectionLossMalInterpIoU',
    # Inner-PIoU: Powerful-IoU (Neural Networks 2024) + Inner-IoU auxiliary box.
    #   inner_piou  -> PIoU v1 + Inner
    #   inner_piou2 -> PIoU v2 (non-monotonic focusing) + Inner
    # ThermalIoU (Li et al., IEEE SPL 2026) + Inner-IoU auxiliary box.
    'thermal_inner_iou': 'ultralytics.utils.loss_plugins.thermal_inner_iou_loss.RTDETRDetectionLossThermalInnerIoU',
    'inner_piou': 'ultralytics.utils.loss_plugins.inner_piou_loss.RTDETRDetectionLossInnerPIoU',
    'inner_piou2': 'ultralytics.utils.loss_plugins.inner_piou_loss.RTDETRDetectionLossInnerPIoU2',
    'inner_piou2': 'ultralytics.utils.loss_plugins.inner_piou_loss.RTDETRDetectionLossInnerPIoU2',
    # ASIoU (Pan et al., Pattern Recognition Letters 2026) + Inner-IoU auxiliary box:
    # third innovation point for rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7.
    'asiou_inner': 'ultralytics.utils.loss_plugins.asiou_inner_loss.RTDETRDetectionLossASInnerIoU',
    # AIoU (Adaptive IoU with Edge Alignment Penalty) + Inner-IoU auxiliary box:
    'aiou_inner': 'ultralytics.utils.loss_plugins.aiou_inner_loss.RTDETRDetectionLossAInnerIoU',
    # SR-MAL (CVPR 2025 DEIM MAL + scale-robust NWD/IoU blended quality target,
    # + assignment-supervision aligned matcher). gamma/tau/nwd_const/align_matcher
    # via loss_params. tau=0.0 => exact pure-DEIM-MAL probe.
    'sr_mal': 'ultralytics.utils.loss_plugins.sr_mal_loss.RTDETRDetectionLossSRMAL',
    # DSAW (DCNet, Pattern Recognition 2026): dynamic scale-aware supervision
    # weighting. cls positives x (1+beta*lam*eta), geometry x (1-alpha*lam).
    # beta=0 & alpha=0 => bit-exact baseline VFL. Params via loss_params.
    'dsaw': 'ultralytics.utils.loss_plugins.dsaw_loss.RTDETRDetectionLossDSAW',
    # DSAW-v2: sharpened scale focus (lam^p) plus dynamic geometry annealing.
    'dsaw_v2': 'ultralytics.utils.loss_plugins.dsaw_v2_loss.RTDETRDetectionLossDSAWV2',
    # InnerInterpIoU-DSAW3: DSAW-v2 weighting + Inner-InterpIoU geometry metric.
    'inner_interpiou_dsaw3': 'ultralytics.utils.loss_plugins.inner_interpiou_dsaw3_loss.RTDETRDetectionLossInnerInterpIoUDSAW3',
    # DSAW-v4: segmented geometry (IoU-gated GIoU/InnerInterpIoU blend) +
    # decoupled weights (L1 softened, IoU branch unsoftened by default).
    'dsaw_v4': 'ultralytics.utils.loss_plugins.dsaw_v4_loss.RTDETRDetectionLossDSAWV4',
    # DSAW-v5: matching-end intervention - scale-aware Hungarian cost
    # (GIoU/NWD blend gated by the same focused lam as the loss), v4 loss kept.
    'dsaw_v5': 'ultralytics.utils.loss_plugins.dsaw_v5_loss.RTDETRDetectionLossDSAWV5',
    # DSAW-v6: matcher-side intervention redesigned after the v5 negative
    # result - curricula (blend weight scaled by eta), no-overlap gating
    # (NWD only for pairs with IoU < overlap_thresh), and a decoupled
    # matcher-side scale gate (linear lam). v4 loss kept bit-exact.
    # nwd_weight=0 => bit-exact v4.
    'dsaw_v6': 'ultralytics.utils.loss_plugins.dsaw_v6_loss.RTDETRDetectionLossDSAWV6',
    # DSAW-v7: v6 matcher + quality-gated warmup. A matcher-side EMA of the
    # mean assigned-pair IoU gates the NWD blend: intervention is OFF in the
    # noise phase (q_warm_ema=0 -> bit-exact v4 cost), ramps in as
    # assignments become meaningful (q_warm_ema crosses q_warmup), and fades
    # out late via the v6 eta curriculum. nwd_weight=0 => bit-exact v4;
    # q_warmup=0 => bit-exact v6.
    'dsaw_v7': 'ultralytics.utils.loss_plugins.dsaw_v7_loss.RTDETRDetectionLossDSAWV7',
    'xfcw': 'ultralytics.utils.loss_plugins.xfcw_loss.RTDETRDetectionLossXFCW',
    'xcas': 'ultralytics.utils.loss_plugins.xcas_loss.RTDETRDetectionLossXCAS',
    'xgbr': 'ultralytics.utils.loss_plugins.xgbr_loss.RTDETRDetectionLossXGBR',
    'xbbr': 'ultralytics.utils.loss_plugins.xmisc_loss.RTDETRDetectionLossXBBR',
    'xvfla': 'ultralytics.utils.loss_plugins.xmisc_loss.RTDETRDetectionLossXVFLA',
    'xgain': 'ultralytics.utils.loss_plugins.xgain_loss.RTDETRDetectionLossXGain',
    'xdnw': 'ultralytics.utils.loss_plugins.xdnw_loss.RTDETRDetectionLossXDNW',
    'xnegg': 'ultralytics.utils.loss_plugins.xb6_loss.RTDETRDetectionLossXNegG',
    'xls': 'ultralytics.utils.loss_plugins.xb6_loss.RTDETRDetectionLossXLS',
    'xqclamp': 'ultralytics.utils.loss_plugins.xdeep_loss.RTDETRDetectionLossXQClamp',
    'xsl1': 'ultralytics.utils.loss_plugins.xdeep_loss.RTDETRDetectionLossXSL1',
    'xfcwcas': 'ultralytics.utils.loss_plugins.xb8_loss.RTDETRDetectionLossXFCWCAS',
    'xsizeq': 'ultralytics.utils.loss_plugins.xb8_loss.RTDETRDetectionLossXSizeQ',
}

# kwargs every plugin receives by default; `loss_params` from the model yaml
# can override any of them (params are applied last).
_BASE_LOSS_KWARGS = dict(
    use_vfl=True,
    use_sl=False,
    use_emasl=False,
    use_svfl=False,
    use_emasvfl=False,
    use_mal=False,
)

__all__ = ('LOSS_REGISTRY', 'build_loss')


def build_loss(name, nc, params=None):
    """
    Instantiate a registered loss by name.

    Args:
        name (str): Key of `LOSS_REGISTRY` (value of `loss_name` in the yaml).
        nc (int): Number of classes.
        params (dict | None): Extra kwargs from `loss_params` in the yaml;
            forwarded to the plugin constructor on top of the base kwargs.

    Returns:
        (nn.Module): The loss criterion instance.
    """
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss_name '{name}'. Available: {sorted(LOSS_REGISTRY)}. "
            f'Register new losses in ultralytics/utils/loss_plugins/__init__.py.')
    mod_path, cls_name = LOSS_REGISTRY[name].rsplit('.', 1)
    cls = getattr(importlib.import_module(mod_path), cls_name)
    kwargs = dict(_BASE_LOSS_KWARGS)
    if params:
        kwargs.update(params)
    return cls(nc=nc, **kwargs)






