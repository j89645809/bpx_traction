"""Public force-estimation contracts and optional PyTorch models."""

from .contracts import WorldForceContract
from .bpx_adapter import BpxForceEstimator

__all__ = ["WorldForceContract", "BpxForceEstimator"]

# Keep contract/preprocessing tooling usable on machines without PyTorch. The
# model classes remain importable from ``force_estimator.model`` and
# ``force_estimator.world_model`` when the optional dependency is installed.
try:  # pragma: no cover - depends on the host environment
    from .model import ForceEstimator
    from .world_model import MissingStateTCN
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__ += ["ForceEstimator", "MissingStateTCN"]
