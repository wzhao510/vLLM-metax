import contextlib
import json
import os
import subprocess
import time

import pytest


try:
    from openai import OpenAI
    from transformers import AutoTokenizer
except ImportError:
    print("Please install 'torch', 'openai' and 'transformers' first.")
    raise SystemExit(1)

from common import (
    dump_file as _dump_file,
    popen_own_group as _popen_own_group,
    query_and_measure_ttft,
    terminate_process_group as _terminate,
    wait_for_free_gpu_memory as _wait_for_free_gpu_memory,
    wait_for_port as _wait_for_port,
)


VLLM_MODEL_PATH = "/mxstorage/pde_ai/models/llm/Qwen/Qwen3-4B"
VLLM_HOST = "localhost"

MAX_MODEL_LEN = 32768
TEST_PROMPT_TOKENS = 30000

EXPECTED_ANSWER_KEYWORDS = ("beijing", "北京")

COORDINATOR_HOST = "localhost"
COORDINATOR_PORT = 9300

# Two independent (server, engine) slots -- "A" populates the cache, "B" is
# the one that must reuse it. Requires 2 GPUs; tests 03/04 use both slots,
# tests 01/02 only use slot A.
SLOT_A = {
    "cuda_device": "1",
    "lmcache_port": 6555,
    "lmcache_http_port": 7555,
    "vllm_port": 8300,
    "instance_id": "lmcache-a",
    "p2p_port": 8555,
}
SLOT_B = {
    "cuda_device": "2",
    "lmcache_port": 6556,
    "lmcache_http_port": 7556,
    "vllm_port": 8301,
    "instance_id": "lmcache-b",
    "p2p_port": 8556,
}


@contextlib.contextmanager
def lmcache_coordinator(tmp_path_factory):
    """Start `lmcache coordinator` (membership only, never touches KV data).

    Only needed for the P2P test -- centralized sharing and plain L1/L2 tests
    don't use a coordinator at all.
    """
    log_file = tmp_path_factory.mktemp("kv_mp_coordinator").joinpath("coordinator.log")
    process = None
    log_handle = None
    try:
        cmd = [
            "lmcache", "coordinator",
            "--host", COORDINATOR_HOST,
            "--port", str(COORDINATOR_PORT),
        ]
        print(f"\n--- Starting lmcache coordinator ---\n{' '.join(cmd)}")
        log_handle = open(log_file, "w")
        process = _popen_own_group(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        _wait_for_port(
            COORDINATOR_HOST, COORDINATOR_PORT, timeout=30,
            process=process, log_paths=log_file, stage="coordinator startup",
        )
        print("lmcache coordinator is ready.")
        yield
    finally:
        _terminate(process, "lmcache coordinator")
        if log_handle:
            log_handle.close()


@contextlib.contextmanager
def lmcache_mp_service(
    name,
    tmp_path_factory,
    *,
    slot,
    l2_base_path=None,
    p2p_peer_slot=None,
):
    """Start one `lmcache server` + one vLLM (LMCacheMPConnector) pair.

    slot: one of SLOT_A / SLOT_B -- picks the GPU and the full set of ports
        for this instance, so two instances can run at once without clashing.
    l2_base_path: if set, registers a dependency-free `fs` L2 adapter here.
        Pass the SAME path to two different slots/invocations to test
        centralized (shared-L2) sharing.
    p2p_peer_slot: if set (also requires a running lmcache_coordinator),
        enables P2P on this server so it can read a peer's L1 directly.
        Leave unset for the plain L1/L2/centralized tests.
    """
    test_run_path = tmp_path_factory.mktemp(f"kv_mp_{name}")
    lmcache_log = test_run_path / "lmcache_server.log"
    vllm_log = test_run_path / "vllm_server.log"

    print(f"\n[Debug] KVCache MP '{name}' logs: {test_run_path}")

    env = os.environ.copy()
    # The lmcache server needs the same GPU visible as vLLM: on the default
    # (CUDA IPC) transfer path it maps vLLM's KV cache memory directly via
    # cudaIpcOpenMemHandle, which requires both processes to agree on which
    # physical device is "device 0".
    env["CUDA_VISIBLE_DEVICES"] = slot["cuda_device"]
    # Required on this MetaX (MACA) GPU platform: without it, the lmcache
    # server and vLLM worker cannot share GPU memory across processes and
    # the REGISTER_KV_CACHE handshake hangs indefinitely.
    env["MACA_MPS_MODE"] = "1"

    lmcache_process = None
    vllm_process = None
    lmcache_log_handle = None
    vllm_log_handle = None
    try:
        print(f"\n--- Starting lmcache server ({name}) ---")
        lmcache_cmd = [
            "lmcache", "server",
            "--host", "0.0.0.0" if p2p_peer_slot else "localhost",
            "--port", str(slot["lmcache_port"]),
            "--http-port", str(slot["lmcache_http_port"]),
            "--chunk-size", "256",
            "--l1-size-gb", "5",
            "--eviction-policy", "LRU",
        ]
        if l2_base_path is not None:
            l2_base_path.mkdir(parents=True, exist_ok=True)
            lmcache_cmd += [
                "--l2-adapter",
                json.dumps({"type": "fs", "base_path": str(l2_base_path)}),
            ]
        if p2p_peer_slot is not None:
            lmcache_cmd += [
                "--l1-align-bytes", "65536",
                "--instance-id", slot["instance_id"],
                "--coordinator-url", f"http://{COORDINATOR_HOST}:{COORDINATOR_PORT}",
                "--coordinator-advertise-ip", "127.0.0.1",
                "--p2p-advertise-url", f"127.0.0.1:{slot['p2p_port']}",
                "--p2p-listen-url", f"0.0.0.0:{slot['p2p_port']}",
            ]
        print(f"lmcache server command: {' '.join(lmcache_cmd)}")
        lmcache_log_handle = open(lmcache_log, "w")
        lmcache_process = _popen_own_group(
            lmcache_cmd, env=env, stdout=lmcache_log_handle, stderr=subprocess.STDOUT
        )
        _wait_for_port(
            "localhost", slot["lmcache_http_port"], timeout=60,
            process=lmcache_process, log_paths=lmcache_log,
            stage=f"{name} lmcache server startup",
        )
        print("lmcache server is ready.")

        print(f"\n--- Starting vLLM ({name}, LMCacheMPConnector) ---")
        kv_transfer_config = {
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": slot["lmcache_port"],
            },
        }
        vllm_cmd = [
            "vllm", "serve", VLLM_MODEL_PATH,
            "--port", str(slot["vllm_port"]),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--no-enable-prefix-caching",
            "--enforce-eager",
            "--kv-transfer-config", json.dumps(kv_transfer_config),
        ]
        print(f"vLLM command: {' '.join(vllm_cmd)}")
        vllm_log_handle = open(vllm_log, "w")
        vllm_process = _popen_own_group(
            vllm_cmd, env=env, stdout=vllm_log_handle, stderr=subprocess.STDOUT
        )
        print("Waiting for vLLM service to load the model...")
        _wait_for_port(
            VLLM_HOST, slot["vllm_port"], timeout=180,
            process=vllm_process, log_paths=vllm_log,
            stage=f"{name} vllm startup",
        )
        print("vLLM service is ready.")

        yield {
            "port": slot["vllm_port"],
            "host": VLLM_HOST,
            "log_file": vllm_log,
            "lmcache_log_file": lmcache_log,
            "type": name,
        }

    finally:
        print(f"\n--- Stopping vLLM + lmcache server ({name}) ---")
        # Stop the client (vLLM) before the server so it does not spend its
        # shutdown window retrying a connection to an already-dead server.
        _terminate(vllm_process, f"vLLM ({name})")
        _terminate(lmcache_process, f"lmcache server ({name})")

        if vllm_log_handle:
            vllm_log_handle.close()
        if lmcache_log_handle:
            lmcache_log_handle.close()

        if vllm_process is not None:
            _wait_for_free_gpu_memory(slot["cuda_device"])


def _get_long_prompt():
    print(f"Loading tokenizer from {VLLM_MODEL_PATH} to build a long prompt...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_PATH)
    except Exception as e:
        print(f"Warning: failed to load local tokenizer, fallback to gpt2. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    filler_sentence = (
        "The quick brown fox jumps over the lazy dog while the LMCache "
        "key-value store keeps every previously computed token ready for "
        "instant reuse. "
    )
    tokens_per_repeat = len(tokenizer.encode(filler_sentence))
    repeats_needed = (TEST_PROMPT_TOKENS // tokens_per_repeat) + 1
    long_context = filler_sentence * repeats_needed
    question = (
        "\n\nIgnore all of the filler text above; it is meaningless padding. "
        "What is the capital city of China? Answer with only the city name, "
        "nothing else."
    )
    prompt = long_context + question

    final_tokens = tokenizer.encode(prompt)
    if len(final_tokens) > MAX_MODEL_LEN:
        print(f"Warning: prompt too long ({len(final_tokens)}), truncating to fit {MAX_MODEL_LEN}.")
        # Trim from the FRONT (keep the tail) so the question at the very
        # end of the prompt is never the part that gets cut off.
        prompt = tokenizer.decode(final_tokens[-(MAX_MODEL_LEN - 50):])
        final_tokens = tokenizer.encode(prompt)
    elif len(final_tokens) < TEST_PROMPT_TOKENS * 0.9:
        # Sanity check: catches a repeat of the encode/decode round-trip
        # shrinkage bug above if it ever resurfaces for a different model's
        # tokenizer/filler text.
        print(
            f"Warning: realized prompt ({len(final_tokens)} tokens) is well "
            f"below the {TEST_PROMPT_TOKENS}-token target -- the cold/warm "
            "TTFT gap may be too small to reliably tell apart."
        )

    print(f"Generated prompt tokens: {len(final_tokens)}")
    return prompt, tokenizer


def _client_for(service_info):
    base_url = f"http://{service_info['host']}:{service_info['port']}/v1"
    client = OpenAI(api_key="dummy-key", base_url=base_url)
    try:
        models = client.models.list()
        model_id = models.data[0].id
        print(f"Connected to vLLM service, model ID: {model_id}")
    except Exception as e:
        _dump_file(service_info["log_file"])
        pytest.fail(f"Cannot connect to vLLM service: {e}. Check log: {service_info['log_file']}")
    return client, model_id


def _report_and_assert(tier_name, cold_ttft, warm_ttft, cold_text, warm_text):
    improvement = cold_ttft - warm_ttft
    speedup_factor = cold_ttft / warm_ttft
    print("\n" + "=" * 30)
    print(f"KVCache MP {tier_name} offload result:")
    print(f"  Cold cache TTFT: {cold_ttft:.3f}s")
    print(f"  Warm cache TTFT: {warm_ttft:.3f}s")
    print(f"  TTFT improvement: {improvement:.3f}s ({speedup_factor:.1f}x faster)")
    print(f"  Cold text: {cold_text!r}")
    print(f"  Warm text: {warm_text!r}")
    print("=" * 30)


    assert _contains_expected_answer(cold_text), (
        f"KVCache MP {tier_name}: cold response did not contain the expected "
        f"answer {EXPECTED_ANSWER_KEYWORDS}. cold={cold_text!r}"
    )
    assert _contains_expected_answer(warm_text), (
        f"KVCache MP {tier_name}: cache-hit response did not contain the "
        f"expected answer {EXPECTED_ANSWER_KEYWORDS} -- possible cache "
        f"corruption. warm={warm_text!r}"
    )

    assert warm_ttft < cold_ttft, (
        f"Warm cache ({warm_ttft:.3f}s) was not faster than "
        f"cold cache ({cold_ttft:.3f}s)."
    )


def _contains_expected_answer(text):
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in EXPECTED_ANSWER_KEYWORDS)


def test_01_kvcache_mp_l1_offload(tmp_path_factory):
    """Single lmcache server, no L2 adapter: warm hits must come from L1."""
    with lmcache_mp_service("l1", tmp_path_factory, slot=SLOT_A) as service_info:
        print("\n--- Testing KVCache MP L1 offload ---")
        client, model_id = _client_for(service_info)
        prompt, _ = _get_long_prompt()

        cold_ttft, cold_text = query_and_measure_ttft(client, model_id, prompt)
        print(f"\nCold cache TTFT: {cold_ttft:.3f}s")
        assert cold_ttft > 0, "Cold cache query was unexpectedly too fast."

        time.sleep(1)
        print("\n--- Warm cache query (same lmcache server, same L1) ---")
        warm_ttft, warm_text = query_and_measure_ttft(client, model_id, prompt)
        print(f"\nWarm cache TTFT: {warm_ttft:.3f}s")

        _report_and_assert("L1", cold_ttft, warm_ttft, cold_text, warm_text)


def test_02_kvcache_mp_l2_offload(tmp_path_factory):
    """Two SEPARATE lmcache-server + vLLM lifecycles (same slot, restarted)
    sharing one `fs` L2 adapter directory. The second lmcache server starts
    with an empty L1 (fresh process), so any speedup on the second run can
    only come from L2 persistence -- "engine restarts, KV cache survives".
    """
    l2_base_path = tmp_path_factory.mktemp("kv_mp_l2_data")

    print("\n--- Testing KVCache MP L2 offload: cold phase (fresh L1 + L2) ---")
    with lmcache_mp_service("l2-cold", tmp_path_factory, slot=SLOT_A, l2_base_path=l2_base_path) as service_info:
        client, model_id = _client_for(service_info)
        prompt, _ = _get_long_prompt()

        cold_ttft, cold_text = query_and_measure_ttft(client, model_id, prompt)
        print(f"\nCold cache TTFT: {cold_ttft:.3f}s")
        assert cold_ttft > 0, "Cold cache query was unexpectedly too fast."

        # StoreController pushes L1 -> L2 asynchronously; give it time to
        # finish flushing before this lmcache server process is torn down.
        print("Waiting for the async L1->L2 flush to complete...")
        time.sleep(5)

    # Give the OS a moment to release the ports before the second server
    # binds them again.
    time.sleep(2)

    print("\n--- Testing KVCache MP L2 offload: warm phase (fresh L1, same L2 dir) ---")
    with lmcache_mp_service("l2-warm", tmp_path_factory, slot=SLOT_A, l2_base_path=l2_base_path) as service_info:
        client, model_id = _client_for(service_info)
        prompt, _ = _get_long_prompt()

        warm_ttft, warm_text = query_and_measure_ttft(client, model_id, prompt)
        print(f"\nWarm cache TTFT: {warm_ttft:.3f}s")

        _report_and_assert("L2", cold_ttft, warm_ttft, cold_text, warm_text)


def test_03_kvcache_mp_centralized_sharing(tmp_path_factory):
    """Two DIFFERENT, SIMULTANEOUSLY-running (lmcache server, vLLM) pairs on
    two GPUs, both L2 adapters pointed at the SAME `fs` directory. Instance A
    populates the cache; instance B -- which never saw this prompt and has an
    empty L1 of its own -- must read it back through the shared L2, proving
    genuine cross-instance centralized sharing (not just single-instance L2
    persistence, which test_02 already covers).
    """
    l2_base_path = tmp_path_factory.mktemp("kv_mp_centralized_l2_data")

    with lmcache_mp_service("central-a", tmp_path_factory, slot=SLOT_A, l2_base_path=l2_base_path) as svc_a:
        client_a, model_id_a = _client_for(svc_a)
        prompt, _ = _get_long_prompt()

        print("\n--- Instance A: cold query, populates shared L2 ---")
        cold_ttft, cold_text = query_and_measure_ttft(client_a, model_id_a, prompt)
        print(f"\nInstance A cold TTFT: {cold_ttft:.3f}s")
        assert cold_ttft > 0, "Instance A's cold query was unexpectedly too fast."

        print("Waiting for the async L1->L2 flush to complete...")
        time.sleep(5)

        with lmcache_mp_service("central-b", tmp_path_factory, slot=SLOT_B, l2_base_path=l2_base_path) as svc_b:
            client_b, model_id_b = _client_for(svc_b)

            print("\n--- Instance B: same prompt, own L1 is empty, must hit shared L2 ---")
            warm_ttft, warm_text = query_and_measure_ttft(client_b, model_id_b, prompt)
            print(f"\nInstance B TTFT (via shared L2): {warm_ttft:.3f}s")

            _report_and_assert("centralized-sharing", cold_ttft, warm_ttft, cold_text, warm_text)


def test_04_kvcache_mp_p2p_sharing(tmp_path_factory):
    """A coordinator + two P2P-enabled (lmcache server, vLLM) pairs on two
    GPUs, with NO shared L2 at all. Instance A populates only its own local
    L1; instance B has no L2 fallback, so a hit can only come from a direct
    P2P read of instance A's L1 over the transfer-channel (NIXL by default,
    loopback/TCP on a single node -- functional check only, not a latency
    benchmark; see docs/source/mp/p2p.rst for real RDMA-fabric numbers).
    """
    with lmcache_coordinator(tmp_path_factory):
        with lmcache_mp_service("p2p-a", tmp_path_factory, slot=SLOT_A, p2p_peer_slot=SLOT_B) as svc_a:
            client_a, model_id_a = _client_for(svc_a)
            prompt, _ = _get_long_prompt()

            print("\n--- Instance A: cold query, populates only its own local L1 ---")
            cold_ttft, cold_text = query_and_measure_ttft(client_a, model_id_a, prompt)
            print(f"\nInstance A cold TTFT: {cold_ttft:.3f}s")
            assert cold_ttft > 0, "Instance A's cold query was unexpectedly too fast."

            # Give the P2P controller a moment to discover the peer via the
            # coordinator before instance B even starts (avoids racing the
            # ~5s discovery poll interval against B's first lookup).
            time.sleep(6)

            with lmcache_mp_service("p2p-b", tmp_path_factory, slot=SLOT_B, p2p_peer_slot=SLOT_A) as svc_b:
                client_b, model_id_b = _client_for(svc_b)

                print("\n--- Instance B: same prompt, no L2, must P2P-read from A ---")
                warm_ttft, warm_text = query_and_measure_ttft(client_b, model_id_b, prompt)
                print(f"\nInstance B TTFT (via P2P): {warm_ttft:.3f}s")

                _report_and_assert("p2p-sharing", cold_ttft, warm_ttft, cold_text, warm_text)
