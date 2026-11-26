import json
import os
import subprocess
import textwrap
import time
from pathlib import Path

import pytest
import requests

from common import (
    dump_logs as _dump_process_logs,
    popen_own_group,
    terminate_all,
    wait_for_free_gpu_memory,
    wait_for_port as _wait_for_port,
)


TEST_DIR = Path(__file__).resolve().parent

VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-32B/"
HOST = "localhost"
PROXY_PORT = 9487
PREFILLER_PORT = 7100
DECODER1_PORT = 7200
DECODER2_PORT = 7201

DECODER1_INIT_PORTS = [7300, 7301]
DECODER1_ALLOC_PORTS = [7400, 7401]
DECODER2_INIT_PORTS = [7302, 7303]
DECODER2_ALLOC_PORTS = [7402, 7403]

PROXY_INTERNAL_PORT = 7500
PROXY_SCRIPT_PATH = TEST_DIR / "disagg_proxy_lmcache_server.py"
FIXED_BUFFER_SIZE = 3355443200
GPU_MEMORY_UTILIZATION = 0.85


@pytest.fixture(scope="module")
def cluster_1p2d_tp2(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run_1p2d_tp2")
    print(f"\n[Info] Test artifacts dir: {run_dir}")

    p_conf = run_dir / "prefiller.yaml"
    p_conf.write_text(textwrap.dedent(f"""
        local_cpu: False
        max_local_cpu_size: 0
        max_local_disk_size: 0
        enable_pd: True
        transfer_channel: "nixl"
        pd_role: "sender"
        pd_proxy_host: "{HOST}"
        pd_proxy_port: {PROXY_INTERNAL_PORT}
        pd_buffer_size: {FIXED_BUFFER_SIZE}
        pd_buffer_device: "cuda"
        nixl_backends: [UCX]
    """))

    d1_conf = run_dir / "decoder1.yaml"
    d1_conf.write_text(textwrap.dedent(f"""
        local_cpu: False
        max_local_cpu_size: 0
        enable_pd: True
        transfer_channel: "nixl"
        pd_role: "receiver"
        pd_peer_host: "{HOST}"
        pd_peer_init_port: {json.dumps(DECODER1_INIT_PORTS)}
        pd_peer_alloc_port: {json.dumps(DECODER1_ALLOC_PORTS)}
        pd_buffer_size: {FIXED_BUFFER_SIZE}
        pd_buffer_device: "cuda"
        nixl_backends: [UCX]
    """))

    d2_conf = run_dir / "decoder2.yaml"
    d2_conf.write_text(textwrap.dedent(f"""
        local_cpu: False
        max_local_cpu_size: 0
        enable_pd: True
        transfer_channel: "nixl"
        pd_role: "receiver"
        pd_peer_host: "{HOST}"
        pd_peer_init_port: {json.dumps(DECODER2_INIT_PORTS)}
        pd_peer_alloc_port: {json.dumps(DECODER2_ALLOC_PORTS)}
        pd_buffer_size: {FIXED_BUFFER_SIZE}
        pd_buffer_device: "cuda"
        nixl_backends: [UCX]
    """))

    processes = {}
    logs = {}
    log_paths = {
        "decoder1": run_dir / "decoder1.log",
        "decoder2": run_dir / "decoder2.log",
        "prefiller": run_dir / "prefiller.log",
        "proxy": run_dir / "proxy.log",
    }

    base_env = os.environ.copy()
    base_env.update({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "UCX_TLS": "maca_ipc,maca_copy,tcp",
        "UCX_PROTO_ENABLE": "y",
        "LMCACHE_USE_EXPERIMENTAL": "True",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
    })

    try:
        logs["d1"] = open(log_paths["decoder1"], "w")
        d1_env = base_env.copy()
        d1_env["CUDA_VISIBLE_DEVICES"] = "2,3"
        d1_env["LMCACHE_CONFIG_FILE"] = str(d1_conf)
        d1_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(DECODER1_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", "qwen2.5",
            "--no-enable-prefix-caching",
            "--kv-transfer-config",
            json.dumps({
                "kv_connector": "LMCacheConnectorV1",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "discard_partial_chunks": False,
                    "lmcache_rpc_port": "consumer1",
                    "skip_last_n_tokens": 1,
                },
            }),
        ]
        processes["decoder1"] = popen_own_group(d1_cmd, env=d1_env, stdout=logs["d1"], stderr=subprocess.STDOUT)

        logs["d2"] = open(log_paths["decoder2"], "w")
        d2_env = base_env.copy()
        d2_env["CUDA_VISIBLE_DEVICES"] = "4,5"
        d2_env["LMCACHE_CONFIG_FILE"] = str(d2_conf)
        d2_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(DECODER2_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", "qwen2.5",
            "--no-enable-prefix-caching",
            "--kv-transfer-config",
            json.dumps({
                "kv_connector": "LMCacheConnectorV1",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "discard_partial_chunks": False,
                    "lmcache_rpc_port": "consumer2",
                    "skip_last_n_tokens": 1,
                },
            }),
        ]
        processes["decoder2"] = popen_own_group(d2_cmd, env=d2_env, stdout=logs["d2"], stderr=subprocess.STDOUT)

        logs["pre"] = open(log_paths["prefiller"], "w")
        p_env = base_env.copy()
        p_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        p_env["LMCACHE_CONFIG_FILE"] = str(p_conf)
        p_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(PREFILLER_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "2",
            "--served-model-name", "qwen2.5",
            "--no-enable-prefix-caching",
            "--kv-transfer-config",
            json.dumps({
                "kv_connector": "LMCacheConnectorV1",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "discard_partial_chunks": False,
                    "lmcache_rpc_port": "producer1",
                },
            }),
        ]
        processes["prefiller"] = popen_own_group(p_cmd, env=p_env, stdout=logs["pre"], stderr=subprocess.STDOUT)

        _wait_for_port(HOST, DECODER1_PORT, 300, processes=processes, log_paths=log_paths, log_handles=logs, stage="decoder1 startup")
        _wait_for_port(HOST, DECODER2_PORT, 300, processes=processes, log_paths=log_paths, log_handles=logs, stage="decoder2 startup")
        _wait_for_port(HOST, PREFILLER_PORT, 300, processes=processes, log_paths=log_paths, log_handles=logs, stage="prefiller startup")

        logs["proxy"] = open(log_paths["proxy"], "w")
        init_ports_str = ",".join(map(str, DECODER1_INIT_PORTS))
        alloc_ports_str = ",".join(map(str, DECODER1_ALLOC_PORTS))
        proxy_cmd = [
            "python3", str(PROXY_SCRIPT_PATH),
            "--host", HOST, "--port", str(PROXY_PORT),
            "--prefiller-host", HOST, "--prefiller-port", str(PREFILLER_PORT), "--num-prefillers", "1",
            "--decoder-host", HOST, "--decoder-port", str(DECODER1_PORT),
            "--decoder-init-port", init_ports_str,
            "--decoder-alloc-port", alloc_ports_str,
            "--proxy-host", HOST, "--proxy-port", str(PROXY_INTERNAL_PORT),
            "--num-decoders", "2",
        ]
        processes["proxy"] = popen_own_group(proxy_cmd, stdout=logs["proxy"], stderr=subprocess.STDOUT)
        _wait_for_port(HOST, PROXY_PORT, 60, processes=processes, log_paths=log_paths, log_handles=logs, stage="proxy startup")
        yield run_dir

    finally:
        # Signal every process's whole group (proxy + the 3 vLLM instances)
        # together -- waiting out the proxy's full teardown timeout before
        # even signalling the heavy GPU-holding vLLM processes needlessly
        # delays their shutdown/reclaim.
        terminate_all(processes)

        for f in logs.values():
            f.close()

        # Confirm the GPUs used by prefiller (0,1) / decoder1 (2,3) /
        # decoder2 (4,5) are actually free before the next test starts --
        # terminate_all only confirms the processes are gone, not that the
        # driver has finished reclaiming their memory (can lag by tens of
        # seconds on this platform).
        for cuda_device in range(6):
            wait_for_free_gpu_memory(cuda_device)


def send_chat_request(request_id, prompt):
    url = f"http://{HOST}:{PROXY_PORT}/v1/chat/completions"
    payload = {
        "model": "qwen2.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0,
        "stream": True,
    }

    collected_messages = []
    received_done = False
    try:
        with requests.post(url, json=payload, timeout=30, stream=True) as resp:
            if resp.status_code != 200:
                return False, f"Status code {resp.status_code}"

            for line in resp.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    json_str = decoded_line[5:].strip()
                    if json_str == "[DONE]":
                        received_done = True
                        break
                    try:
                        data = json.loads(json_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                collected_messages.append(content)
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return False, str(e)

    full_text = "".join(collected_messages)
    if not received_done:
        return False, "Stream not finished (No [DONE])"
    if not full_text:
        return False, "Empty response"
    return True, full_text


def test_chat_completion_round_robin(cluster_1p2d_tp2):
    test_cases = [
        {"prompt": "1 + 1 = ?", "expect": "2"},
        {"prompt": "Say exactly the word: Hello", "expect": "Hello"},
    ]

    failures = []
    for i, case in enumerate(test_cases):
        req_id = i + 1
        success, result = send_chat_request(req_id, case["prompt"])
        if not success:
            failures.append(f"Req #{req_id} Protocol Failed: {result}")
            continue
        if case["expect"].lower() not in result.lower():
            failures.append(f"Req #{req_id} Answer mismatch. Expected '{case['expect']}', Got '{result}'")
        time.sleep(3.0)

    if failures:
        pytest.fail(f"Test Failed ({len(failures)} failures):\n" + "\n".join(failures))

