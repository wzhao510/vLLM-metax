# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Optimize data-parallel coordinator load balancing for MetaX.
#
# Affected versions: v0.21.0
# -----------------------------------------------


from vllm.logger import init_logger

from vllm.v1.engine.coordinator import DPCoordinatorProc
from vllm_metax.patch.utils import patch

logger = init_logger(__name__)


@patch("vllm.v1.engine.coordinator", "DPCoordinatorProc.run_coordinator")
def run_coordinator(
    engine_count: int,
    front_publish_address: str,
    back_output_address: str,
    back_publish_address: str,
    zmq_addr_pipe=None,
    min_stats_update_interval_ms: int = 100,
    enable_wave_coordination: bool = True,
):
    # /------------------------  Metax Modification -------------------------\
    coordinator = DPCoordinatorProc(
        # \------------------------- Metax Modification -------------------------/
        engine_count=engine_count,
        min_stats_update_interval_ms=min_stats_update_interval_ms,
        enable_wave_coordination=False,
    )
    try:
        coordinator.process_input_socket(
            front_publish_address,
            back_output_address,
            back_publish_address,
            zmq_addr_pipe,
        )
    except KeyboardInterrupt:
        logger.info("DP Coordinator process exiting")
    finally:
        if zmq_addr_pipe is not None:
            zmq_addr_pipe.close()
