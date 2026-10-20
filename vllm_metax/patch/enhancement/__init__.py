# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Load MetaX enhancement patches and registration hooks at plugin initialization.
#     This initializer applies the patches by importing the child modules.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
#
# Remove at: All enhancement patches imported here have been removed or migrated to
#     native extension points.
# -----------------------------------------------------------------------------

# module level imports

from . import chores  # noqa: F401
from . import distributed  # noqa: F401
from . import quant_kernels  # noqa: F401
from . import joyai_support  # noqa: F401

# single files
from . import utils  # noqa: F401

from . import dbo  # noqa: F401
from . import glm5next_flash_patches  # noqa: F401 — GLM-5.3 Flash 适配
    # 含两块，互不依赖但成套出现：
    #   1. compressed_tensors MoE scheme 分组路由 → RoutedExperts 绑定 FusedMoE/W4A8
    #      （并在 RoutedExperts 构造时打印实际选中的 quant_method）
    #   2. QuaRot 在线逆旋转 pre-hook → 注入最终 RMSNorm
# -----------------------------------------------------------------------------
# 旧的拆分版 patch 文件保留作为备份，待新文件验证通过后删除：
#   from . import glm5next_rotation           # ← 已并入 glm5next_flash_patches
#   from . import compressed_tensors_moe_scheme   # ← 已并入
#   from . import glm5next_moe_params_mapping # ← 已废弃（.routed_experts 是真实层级）
# -----------------------------------------------------------------------------

# 诊断开关（排 KeyError 时临时打开，正常运行注释掉）：
# from . import glm5next_moe_debug  # noqa: F401 — dump MoE param names on KeyError
