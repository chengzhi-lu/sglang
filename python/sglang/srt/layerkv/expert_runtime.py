"""LayerKV expert runtime mixin aggregation."""

from __future__ import annotations

if __package__:
    from .expert_policy import LayerKVExpertPolicyMixin
    from .expert_planner import LayerKVExpertPlannerMixin
    from .expert_install import LayerKVExpertInstallMixin
    from .expert_backing import LayerKVExpertBackingMixin
    from .expert_hooks import LayerKVExpertHooksMixin
else:  # pragma: no cover - direct file-loading smoke tests.
    from expert_policy import LayerKVExpertPolicyMixin
    from expert_planner import LayerKVExpertPlannerMixin
    from expert_install import LayerKVExpertInstallMixin
    from expert_backing import LayerKVExpertBackingMixin
    from expert_hooks import LayerKVExpertHooksMixin


class LayerKVExpertMixin(
    LayerKVExpertPolicyMixin,
    LayerKVExpertPlannerMixin,
    LayerKVExpertInstallMixin,
    LayerKVExpertBackingMixin,
    LayerKVExpertHooksMixin,
):
    pass
