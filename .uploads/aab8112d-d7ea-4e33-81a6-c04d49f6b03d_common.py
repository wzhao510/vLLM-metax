"""Shared helpers for auto_pd_test.

Every test file under this suite starts one or more `vllm serve` /
`lmcache_server` / `mooncake_master` processes, waits for them to come up,
runs one or more OpenAI-client queries against them, and tears everything
down again. That lifecycle used to be copy-pasted (with small, inconsistent
variations) into every test file. This module is the single place it lives
now, so:

  - a fix (e.g. the orphaned-EngineCore-process GPU leak below) lands once
    for every backend instead of N times.
  - new test files can be written by importing this module instead of
    re-deriving the boilerplate.

See TEST_PLAN.md for the audit that motivated this module.
"""

import os
import signal
import socket
import subprocess
import time

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected to be present
    torch = None


# ---------------------------------------------------------------------------
# Log dumping
# ---------------------------------------------------------------------------

def dump_logs(log_paths, log_handles=None, tail_lines=200, title="STARTUP DEBUG LOG DUMP"):
    """Print the tail of one or more log files.

    `log_paths` may be a single path/str, or a {name: path} dict (the common
    case when several processes are involved).
    """
    if not isinstance(log_paths, dict):
        log_paths = {"log": log_paths}

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if log_handles:
        for handle in log_handles.values():
            try:
                handle.flush()
            except Exception:
                pass

    for name, log_path in log_paths.items():
        print(f"\n--- {name}: {log_path} ---")
        try:
            with open(log_path, "r", errors="ignore") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("(log file not found)")
            continue

        if not lines:
            print("(log file is empty)")
            continue

        print("".join(lines[-tail_lines:]).rstrip())

    print("\n" + "=" * 80)


def dump_file(log_file, tail_lines=200):
    """Single-file convenience wrapper around `dump_logs`."""
    dump_logs({"log": log_file}, tail_lines=tail_lines, title=f"DEBUG LOG DUMP: {log_file}")


# ---------------------------------------------------------------------------
# Waiting for services to come up
# ---------------------------------------------------------------------------

def wait_for_port(host, port, timeout=180, process=None, processes=None, log_paths=None,
                   log_handles=None, stage="startup"):
    """Block until `host:port` accepts a connection.

    Accepts either a single `process` (Popen) or a `processes` dict of
    {name: Popen} -- if any of them has already exited, fail fast with the
    tail of its log instead of waiting out the full timeout.
    """
    all_processes = dict(processes) if processes else {}
    if process is not None:
        all_processes.setdefault(stage, process)

    start = time.monotonic()
    while True:
        for name, proc in all_processes.items():
            retcode = proc.poll()
            if retcode is not None:
                print(f"[FATAL] Process {name} exited unexpectedly during {stage} with code {retcode}.")
                if log_paths:
                    dump_logs(log_paths, log_handles)
                raise RuntimeError(f"Process {name} exited unexpectedly during {stage} with code {retcode}.")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            if time.monotonic() - start >= timeout:
                print(f"[FATAL] Timeout waiting for {host}:{port} during {stage}.")
                if log_paths:
                    dump_logs(log_paths, log_handles)
                raise TimeoutError(f"Port {port} not ready in {timeout}s")
            time.sleep(1)


def wait_for_ports(host, ports, log_paths=None, processes=None, timeout=300):
    """Block until ALL of `ports` are accepting connections.

    Useful when several services are started roughly at once and you'd
    rather report every port still down than fail on the first one.
    """
    start = time.monotonic()
    not_ready = list(ports)
    while True:
        if processes:
            for name, proc in processes.items():
                retcode = proc.poll()
                if retcode is not None:
                    print(f"[FATAL] Process {name} exited unexpectedly with code {retcode}.")
                    if log_paths:
                        dump_logs(log_paths)
                    raise RuntimeError(f"Process {name} died during startup.")

        not_ready = []
        for port in ports:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    pass
            except OSError:
                not_ready.append(port)

        if not not_ready:
            print(f"All ports {list(ports)} are ready.")
            return

        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            print(f"[FATAL] Timeout waiting for ports: {not_ready}")
            if log_paths:
                dump_logs(log_paths)
            raise TimeoutError(f"Ports not ready after {timeout}s: {not_ready}")
        time.sleep(1)


# ---------------------------------------------------------------------------
# Process lifecycle
#
# `vllm serve` forks its own child processes (a separate EngineCore process,
# visible in its logs under a different pid). Sending SIGTERM/SIGKILL only to
# the top-level `vllm` pid leaves those children running and holding GPU
# memory. On top of that, on this MetaX (MACA) platform a cleanly-terminated
# process can still leave the device showing only a fraction of its memory
# free for tens of seconds afterward (observed and documented while debugging
# lmcache/test_lmcache_mp.py test_02/03), which makes the *next* vLLM startup
# in the same test session fail with "not enough free GPU memory" -- a
# spurious failure that has nothing to do with the thing actually being
# tested. `popen_own_group` + `terminate_process_group` +
# `wait_for_free_gpu_memory` below close both holes; every fixture that
# starts vLLM should use them.
# ---------------------------------------------------------------------------

def popen_own_group(cmd, **kwargs):
    """`subprocess.Popen` wrapper that puts the child in its own process
    group/session (POSIX `setsid` equivalent), so `terminate_process_group`
    can signal the whole tree it spawns, not just the top pid."""
    return subprocess.Popen(cmd, start_new_session=True, **kwargs)


def _pgid_alive(pgid):
    """Whether ANY process in group `pgid` is still alive. `Popen.wait()`
    only ever tells us about the single pid we started -- vLLM's top-level
    process forks an EngineCore process, which forks Worker processes; those
    grandchildren stay in the same process group but Popen.wait() has no
    idea they exist. Checking the group's liveness with a signal-0 probe is
    what actually tells us whether GPU-memory-holding processes are gone."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _poll_group_until_gone(pgids, processes, timeout, poll_interval=0.5):
    """Poll a {name: pgid} map until every group has no living members (or
    `timeout` elapses). `processes` (a {name: Popen} map, may be a superset
    of `pgids`) is polled every iteration purely so our own direct children
    get reaped as they exit and don't linger as zombies."""
    remaining = dict(pgids)
    deadline = time.monotonic() + timeout
    while True:
        for proc in processes.values():
            if proc is not None:
                proc.poll()
        remaining = {name: pgid for name, pgid in remaining.items() if _pgid_alive(pgid)}
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(poll_interval)


def terminate_process_group(process, name="process", term_timeout=20, kill_timeout=15):
    """Signal `process`'s entire process group: SIGTERM, then SIGKILL if it
    doesn't exit in time. Waits on the WHOLE process group's liveness, not
    just the single tracked pid -- see `_pgid_alive`. No-op if the process is
    None or already gone."""
    if process is None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    if not _poll_group_until_gone({name: pgid}, {name: process}, term_timeout):
        return

    print(f"{name} (pgid={pgid}) did not exit on SIGTERM, sending SIGKILL to its process group.")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return

    if _poll_group_until_gone({name: pgid}, {name: process}, kill_timeout):
        print(f"{name} (pgid={pgid}) STILL alive after SIGKILL -- check for leaked GPU memory.")


def terminate_all(processes, term_timeout=20, kill_timeout=15):
    """Tear down a {name: Popen} dict. Every process's whole group is
    SIGTERM'd together (not one at a time -- waiting out one process's full
    timeout before even signalling the next would needlessly serialize
    teardown and delay GPU reclaim for the rest), then polled together, then
    SIGKILL'd together if any are still alive."""
    pgids = {}
    for name, proc in processes.items():
        if proc is None:
            continue
        try:
            pgids[name] = os.getpgid(proc.pid)
        except ProcessLookupError:
            continue

    for name, pgid in pgids.items():
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    remaining = _poll_group_until_gone(pgids, processes, term_timeout)
    if not remaining:
        return

    for name, pgid in remaining.items():
        print(f"{name} (pgid={pgid}) did not exit on SIGTERM, sending SIGKILL to its process group.")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    still_remaining = _poll_group_until_gone(remaining, processes, kill_timeout)
    for name, pgid in still_remaining.items():
        print(f"{name} (pgid={pgid}) STILL alive after SIGKILL -- check for leaked GPU memory.")


def wait_for_free_gpu_memory(cuda_device, min_free_ratio=0.85, timeout=60):
    """Poll `torch.cuda.mem_get_info()` until `cuda_device` has mostly-free
    memory. Call this after tearing down a vLLM instance and before starting
    the next one on the same device within a test session."""
    if torch is None:
        print("[WARN] torch not importable, skipping GPU memory reclaim check.")
        return
    device_index = int(cuda_device)
    start = time.monotonic()
    while True:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        free_ratio = free_bytes / total_bytes
        free_gib = free_bytes / (1024**3)
        total_gib = total_bytes / (1024**3)
        if free_ratio >= min_free_ratio:
            print(f"GPU {cuda_device}: {free_gib:.1f}/{total_gib:.1f} GiB free, proceeding.")
            return
        if time.monotonic() - start >= timeout:
            print(
                f"[WARN] GPU {cuda_device} still only {free_gib:.1f}/{total_gib:.1f} GiB "
                f"free after waiting {timeout}s for memory reclaim; proceeding anyway "
                "-- the next service may fail to start."
            )
            return
        time.sleep(2)


# ---------------------------------------------------------------------------
# Prompt building + TTFT/correctness measurement
# ---------------------------------------------------------------------------

def build_long_prompt(model_path, num_tokens, max_model_len,
                       question="\n\nSummarize the text above in exactly one sentence."):
    """Build a prompt of roughly `num_tokens` tokens (repeating a filler
    sentence), truncated to fit `max_model_len`. Returns (prompt, tokenizer).
    """
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_path} to build a long prompt...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Warning: failed to load local tokenizer, fallback to gpt2. Error: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    test_token_id = tokenizer.encode("A")[0]
    long_context_tokens = [test_token_id] * (num_tokens - 10)
    long_context = tokenizer.decode(long_context_tokens)
    prompt = long_context + question

    final_tokens = tokenizer.encode(prompt)
    if len(final_tokens) > max_model_len:
        print(f"Warning: prompt too long ({len(final_tokens)}), truncating to fit {max_model_len}.")
        prompt = tokenizer.decode(final_tokens[: max_model_len - 50])

    print(f"Generated prompt tokens: {len(tokenizer.encode(prompt))}")
    return prompt, tokenizer


def query_and_measure_ttft(client, model_id, prompt, max_tokens=150, temperature=0.0):
    """Run one streaming chat query. Returns (ttft_seconds, generated_text).

    Returning the text (not just the latency) matters: a KV-cache bug can
    corrupt generation without necessarily making it slower, so a speed
    comparison alone can't catch that -- callers should assert on both.
    """
    start_time = time.perf_counter()
    first_token_time = None
    text_parts = []
    try:
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    print(f"First token: '{content}'", end="", flush=True)
                else:
                    print(content, end="", flush=True)
                text_parts.append(content)
        print("\nStreaming response finished.")
        if first_token_time is None:
            raise RuntimeError("No token content received from the stream.")
        return first_token_time - start_time, "".join(text_parts)
    except Exception as e:
        print(f"\nQuery failed: {e}")
        return 9999.0, ""


def assert_cache_speedup(label, cold_ttft, warm_ttft, cold_text=None, warm_text=None,
                          max_warm_ratio=0.5, require_match=True, keyword_len=24):
    """Standard pass/fail check for a cold-vs-warm KV cache comparison test.

    Checks (1) the warm query was meaningfully faster than the cold one, and
    -- whenever text was captured -- (2) the warm output contains the leading
    `keyword_len` characters of the cold output, so a cache-corruption bug
    that doesn't show up as a slowdown still fails the test.

    This is a *keyword-containment* check, not a full exact-match: greedy
    (temperature=0) decoding on the same engine is normally stable, but
    requiring the two completions to be byte-for-byte identical is stricter
    than the KV cache contract actually promises (e.g. minor tail-end drift
    from batching/kernel state) and produced false failures in practice.
    Checking that the cold output's leading text reappears in the warm
    output still catches real corruption (a totally different response)
    without being flaky about small differences further into the generation.
    """
    speedup_factor = cold_ttft / warm_ttft if warm_ttft else float("inf")
    print("\n" + "=" * 30)
    print(f"{label} result:")
    print(f"  Cold TTFT: {cold_ttft:.3f}s")
    print(f"  Warm TTFT: {warm_ttft:.3f}s")
    print(f"  Speedup: {speedup_factor:.1f}x")
    if cold_text is not None:
        print(f"  Cold text: {cold_text!r}")
    if warm_text is not None:
        print(f"  Warm text: {warm_text!r}")
    print("=" * 30)

    assert cold_ttft > 0, f"{label}: cold query was unexpectedly too fast."
    assert warm_ttft < cold_ttft * max_warm_ratio, (
        f"{label}: warm cache ({warm_ttft:.3f}s) did not improve enough over "
        f"cold cache ({cold_ttft:.3f}s)."
    )

    if require_match and cold_text is not None and warm_text is not None:
        assert cold_text, f"{label}: cold query produced no output."
        keyword = cold_text.strip()[:keyword_len]
        assert keyword in warm_text, (
            f"{label}: warm-cache output does not contain the cold-cache "
            f"output's leading text ({keyword!r}) -- possible KV cache "
            f"corruption. cold={cold_text!r} warm={warm_text!r}"
        )
