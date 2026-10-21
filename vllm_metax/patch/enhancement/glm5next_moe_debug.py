# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# TEMPORARY DEBUG PATCH — remove once the GLM5Next weight-name mismatch is
# understood.  It does not change any behaviour: it only wraps every
# `load_weights` defined inside vLLM's glm5next model module and, on KeyError,
# dumps the model's real parameter names so they can be compared with the
# checkpoint key that failed to load.
# -----------------------------------------------------------------------------

import re

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase

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
                missing = str(e.args[0]) if e.args else "?"
                allp = [n for n, _ in self.named_parameters()]
                expert = [n for n in allp if "experts" in n]
                routed = [n for n in allp if "routed_experts" in n]
                logger.error("### [DBG] KeyError in %s.load_weights: %s", _cls, missing)
                logger.error(
                    "### [DBG] params total=%d expert=%d routed_experts=%d",
                    len(allp),
                    len(expert),
                    len(routed),
                )
                for n in expert[:80]:
                    logger.error("### [DBG]   %s", n)

                # Dump attention-related names for the failing layer.
                m = re.search(r"layers\.(\d+)\.self_attn", missing)
                if m:
                    layer_idx = m.group(1)
                    attn_prefix = f"layers.{layer_idx}.self_attn"
                    attn = [n for n in allp if attn_prefix in n]
                    logger.error(
                        "### [DBG] attention params for %s (count=%d):",
                        attn_prefix,
                        len(attn),
                    )
                    for n in attn[:120]:
                        logger.error("### [DBG]   %s", n)

                    # Dump the quant_method selected for each LinearBase child.
                    attn_module = next(
                        (m for n, m in self.named_modules() if attn_prefix in n and n.endswith("self_attn")),
                        None,
                    )
                    if attn_module is not None:
                        logger.error(
                            "### [DBG] attention quant_methods for %s:", attn_prefix
                        )
                        for child_name, child in attn_module.named_modules():
                            if isinstance(child, LinearBase):
                                qm = getattr(child, "quant_method", None)
                                qm_name = type(qm).__name__ if qm is not None else "None"
                                child_params = [p for p, _ in child.named_parameters()]
                                logger.error(
                                    "### [DBG]   %-50s quant_method=%s params=%s",
                                    child_name,
                                    qm_name,
                                    child_params[:6],
                                )

                base = missing.rsplit(".", 1)[0]
                base_params = [n for n in allp if n.startswith(base)]
                if base_params:
                    logger.error("### [DBG] params starting with %r:", base)
                    for n in base_params[:40]:
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
