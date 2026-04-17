"""Weight loader. Phase 1 ships the `dummy` branch only. The other branches
raise NotImplementedError so anyone flipping the YAML prematurely gets a
loud error rather than silent wrong output.

Flipping to `our_real` arrives in Phase 3 via kagglehub. Flipping to
`marccoru_baseline` is a Phase 2 optional bonus requiring a manual Google
Drive download (weights moved to private Drive Aug 2024 — see STATE.md).
"""
import torch
import torch.nn as nn

from backend.core.config import Settings
from backend.ml.model import DualHeadUNetpp


def load_weights(cfg: Settings) -> nn.Module:
    source = cfg.ml.weights_source

    if source == "dummy":
        torch.manual_seed(cfg.ml.dummy_seed)  # cfg.ml.dummy_seed = 42
        model = DualHeadUNetpp(in_channels=cfg.ml.in_channels)
        # Sanity: SMP's in_channels=14 adaptation must not produce a dead
        # (zero-std) first conv. See RESEARCH.md §"SMP in_channels=14 init
        # probe" and wave0-probe-results.md §"Probe 3".
        assert model.backbone.encoder.conv1.weight.std().item() > 1e-4, (
            "conv1 dead-init -- SMP did not adapt weights for in_channels=14. "
            "See RESEARCH.md 'Pitfall 4: SMP in_channels=14 Dead Init'."
        )
        # Bias the mask-head output toward ~0.5 so thresholded inference
        # produces non-empty polygons on random-weight dummy runs. Without
        # this shift, sigmoid(random-logit) is noisy around 0.5 and the
        # threshold+area filter in Plan 05 can drop everything. With
        # bias=0.5 the mean logit sits well above 0, guaranteeing the
        # Plan 05 integration test asserts n > 0 (strict).
        with torch.no_grad():
            model.mask_head.bias.data.fill_(0.5)
        return model.eval()

    if source == "marccoru_baseline":
        raise NotImplementedError(
            "marccoru_baseline weights require manual Google Drive download. "
            "Phase 1 default is 'dummy'. See PITFALLS.md and STATE.md."
        )

    if source == "our_real":
        raise NotImplementedError(
            "our_real weights arrive in Phase 3 via kagglehub. "
            "Flip cfg.ml.weights_source only after Phase 3 training completes."
        )

    raise ValueError(f"Unknown weights_source: {source!r}")
