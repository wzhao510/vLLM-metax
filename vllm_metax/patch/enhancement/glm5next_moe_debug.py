# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# TEMPORARY DEBUG PATCH — remove once the GLM5Next MoE weight-name mismatch is
# understood.  It does not change any behaviour: it only wraps every
# `load_weights` defined inside vLLM's glm5next model module and, on KeyError,
# dumps the model's real expert-related parameter names so they can be compared
# with what `fused_moe_make_expert_params_mapping` produces.
# -----------------------------------------------------------------------------

from vllm.logger import init_logger

logger = init_logger(__name__)


def _install() -> None:
    try:
        import vllm.models.glm5next.nvidia.model as M
    except Exception as exc:  # pragma: no cover - debug helper only
        logger.warning("glm5next_moe_debug: cannot import model module: %s", exc)
        return

    def _wrap(cls) -> None:
        orig = cls.__dict__.get("load_weights")
        if orig is None or getattr(orig, "_metax_dbg", False):
            return

        def wrapped(self, weights, _orig=orig, _cls=cls.__name__):
            try:
                return _orig(self, weights)
            except KeyError as e:
                allp = [n for n, _ in self.named_parameters()]
                expert = [n for n in allp if "experts" in n]
                routed = [n for n in allp if "routed_experts" in n]
                logger.error("### [DBG] KeyError in %s.load_weights: %s", _cls, e)
                logger.error(
                    "### [DBG] params total=%d expert=%d routed_experts=%d",
                    len(allp),
                    len(expert),
                    len(routed),
                )
                for n in expert[:80]:
                    logger.error("### [DBG]   %s", n)
                raise

        wrapped._metax_dbg = True
        setattr(cls, "load_weights", wrapped)

    for _name in dir(M):
        try:
            _obj = getattr(M, _name)
        except Exception:
            continue
        if isinstance(_obj, type) and "load_weights" in getattr(_obj, "__dict__", {}):
            _wrap(_obj)


_install()
