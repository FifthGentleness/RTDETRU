# Ultralytics YOLO 🐛, AGPL-3.0 license
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
      exclusive — `loss_name` selects exactly one.
    - Keep each plugin focused on a single override point (`_get_loss_bbox`,
      `_get_loss_class`, matcher, ...). For orthogonal losses that must be
      stacked, add a dedicated combined plugin class instead of multi-switching
      one class.
"""

import importlib

# name -> full dotted path of the loss class
LOSS_REGISTRY = {
    'sd': 'ultralytics.utils.loss_plugins.sd_loss.RTDETRDetectionLossSD',
    'tri_scale': 'ultralytics.models.utils.tri_scale_loss.TriScaleDetectionLoss',
    'sual': 'ultralytics.utils.loss_plugins.sual_loss.RTDETRDetectionLossSUAL',
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
