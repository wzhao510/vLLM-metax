import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

from common import (
    popen_own_group,
    terminate_all,
    wait_for_free_gpu_memory,
    wait_for_port as _wait_for_port,
)


VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-8B/"
TEST_DIR = Path(__file__).resolve().parent
PROXY_SCRIPT_PATH = TEST_DIR / "disagg_proxy_nixl_server.py"

HOST = "127.0.0.1"
PROXY_PORT = 8192
PREFILLER_PORT = 8100
DECODER1_PORT = 8200
DECODER2_PORT = 8300

NIXL_PORT_PREFILLER = 5600
NIXL_PORT_DECODER1 = 5610
NIXL_PORT_DECODER2 = 5620

GPU_MEMORY_UTILIZATION = 0.85
SERVED_MODEL_NAME = "qwen"


@pytest.fixture(scope="module")
def cluster_1p2d_nixl(tmp_path_factory):
    print("\n[Setup] Starting NIXL 1P2D Cluster (TP=2)...")
    run_dir = tmp_path_factory.mktemp("nixl_1p2d_tp2")
    print(f"[Setup] Logs will be saved to: {run_dir}")

    processes = {}
    logs = {}
    log_paths = {
        "prefiller": run_dir / "prefiller.log",
        "decoder1": run_dir / "decoder1.log",
        "decoder2": run_dir / "decoder2.log",
        "proxy": run_dir / "proxy.log",
    }

    base_env = os.environ.copy()
    base_env["UCX_TLS"] = "maca_ipc,maca_copy,tcp"
    base_env["UCX_PROTO_ENABLE"] = "y"

    kv_config = json.dumps({
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
    })

    try:
        print(f"-> Launching Prefiller (Port {PREFILLER_PORT}, NIXL Side Channel {NIXL_PORT_PREFILLER})...")
        logs["pre"] = open(log_paths["prefiller"], "w")
        p_env = base_env.copy()
        p_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        p_env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(NIXL_PORT_PREFILLER)
        p_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(PREFILLER_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", SERVED_MODEL_NAME,
            "--kv-transfer-config", kv_config,
        ]
        processes["prefiller"] = popen_own_group(p_cmd, env=p_env, stdout=logs["pre"], stderr=subprocess.STDOUT)

        print(f"-> Launching Decoder 1 (Port {DECODER1_PORT}, NIXL Side Channel {NIXL_PORT_DECODER1})...")
        logs["d1"] = open(log_paths["decoder1"], "w")
        d1_env = base_env.copy()
        d1_env["CUDA_VISIBLE_DEVICES"] = "2,3"
        d1_env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(NIXL_PORT_DECODER1)
        d1_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(DECODER1_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", SERVED_MODEL_NAME,
            "--kv-transfer-config", kv_config,
        ]
        processes["decoder1"] = popen_own_group(d1_cmd, env=d1_env, stdout=logs["d1"], stderr=subprocess.STDOUT)

        print(f"-> Launching Decoder 2 (Port {DECODER2_PORT}, NIXL Side Channel {NIXL_PORT_DECODER2})...")
        logs["d2"] = open(log_paths["decoder2"], "w")
        d2_env = base_env.copy()
        d2_env["CUDA_VISIBLE_DEVICES"] = "4,5"
        d2_env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(NIXL_PORT_DECODER2)
        d2_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(DECODER2_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", SERVED_MODEL_NAME,
            "--kv-transfer-config", kv_config,
        ]
        processes["decoder2"] = popen_own_group(d2_cmd, env=d2_env, stdout=logs["d2"], stderr=subprocess.STDOUT)

        print("Waiting for vLLM instances to initialize...")
        _wait_for_port(HOST, PREFILLER_PORT, timeout=300, processes=processes, log_paths=log_paths, log_handles=logs, stage="prefiller startup")
        _wait_for_port(HOST, DECODER1_PORT, timeout=300, processes=processes, log_paths=log_paths, log_handles=logs, stage="decoder1 startup")
        _wait_for_port(HOST, DECODER2_PORT, timeout=300, processes=processes, log_paths=log_paths, log_handles=logs, stage="decoder2 startup")

        print(f"-> Launching Proxy on Port {PROXY_PORT}...")
        logs["proxy"] = open(log_paths["proxy"], "w")
        proxy_cmd = [
            "python3", str(PROXY_SCRIPT_PATH),
            "--port", str(PROXY_PORT),
            "--prefiller-hosts", HOST,
            "--prefiller-ports", str(PREFILLER_PORT),
            "--decoder-hosts", HOST, HOST,
            "--decoder-ports", str(DECODER1_PORT), str(DECODER2_PORT),
        ]
        processes["proxy"] = popen_own_group(proxy_cmd, stdout=logs["proxy"], stderr=subprocess.STDOUT)
        _wait_for_port(HOST, PROXY_PORT, timeout=60, processes=processes, log_paths=log_paths, log_handles=logs, stage="proxy startup")

        print("[Setup] Cluster is READY!")
        yield

    finally:
        print("\n[Teardown] Cleaning up processes...")
        # Signal every process's whole group together (proxy + the 3 vLLM
        # instances) -- see common.terminate_all/terminate_process_group.
        terminate_all(processes)

        for f in logs.values():
            f.close()

        # prefiller uses GPUs 0,1; decoder1 uses 2,3; decoder2 uses 4,5.
        # terminate_all only confirms the processes are gone, not that the
        # driver has finished reclaiming their memory.
        for cuda_device in range(6):
            wait_for_free_gpu_memory(cuda_device)
        print("[Teardown] Done.")


def send_chat_request(request_id, prompt):
    print(f"\n[Req #{request_id}] Sending: '{prompt}'")
    url = f"http://{HOST}:{PROXY_PORT}/v1/chat/completions"

    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0,
        "stream": True,
    }

    collected_text = []
    received_done = False

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}: {resp.text}"

            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")

                if decoded.startswith("data:"):
                    json_str = decoded[5:].strip()
                    if json_str == "[DONE]":
                        received_done = True
                        break
                    try:
                        data = json.loads(json_str)
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            collected_text.append(content)
                    except Exception:
                        pass
    except Exception as e:
        return False, str(e)

    full_response = "".join(collected_text)
    print(f"[Req #{request_id} Result] {full_response}")

    if not received_done:
        return False, "Stream interrupted (No [DONE])"
    return True, full_response


def test_nixl_pd_round_robin(cluster_1p2d_nixl):
    test_cases = [
        {"prompt": "10 + 10 = ?", "expect": "20"},
        {"prompt": "1 + 2 = ?", "expect": "3"},
    ]

    failures = []

    for i, case in enumerate(test_cases):
        req_id = i + 1
        success, result = send_chat_request(req_id, case["prompt"])

        if not success:
            failures.append(f"Req #{req_id} Failed: {result}")
        elif case["expect"] not in result:
            failures.append(f"Req #{req_id} Content Error. Expected '{case['expect']}', Got: '{result}'")
        else:
            print(f"Req #{req_id} Passed")

        time.sleep(2.0)

    if failures:
        pytest.fail("\n".join(failures))

