#!/usr/bin/env python3
"""Sweep ladder-training environment counts and record steady-state throughput.

Each count is a separate process running the same loop as ``train_ladder.py``.
Startup and the first ``--warmup`` iterations are excluded from the medians.
GPU samples are attributed to collection and the PPO update separately.

The reported ``samples_per_s`` is ``num_envs * num_steps_per_env / iteration_s``.
``updates_per_hour`` is how many learning iterations finish per hour. ``batch_size``
is the PPO batch (``num_envs * num_steps_per_env``); a larger count is a different
update, not only a faster one.

Example:
    python train_mimic/scripts/benchmark_ladder_envs.py \\
        --sizes 64 128 256 512 1024 2048 4096 \\
        --device cuda:0 \\
        --out logs/benchmarks/ladder_envs.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

_OVERFLOW_RE = re.compile(r"nconmax|njmax|nefc overflow|constraint.*overflow", re.IGNORECASE)

_GPU_QUERY_FULL = (
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "clocks.sm",
    "clocks.max.sm",
    "power.draw",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_thermal_slowdown",
)
_GPU_FIELDS_FULL = (
    "sm_util",
    "mem_util",
    "mem_used_mib",
    "mem_total_mib",
    "sm_clock_mhz",
    "sm_clock_max_mhz",
    "power_w",
    "sw_power_cap",
    "hw_thermal",
)
_GPU_QUERY_BASIC = _GPU_QUERY_FULL[:7]
_GPU_FIELDS_BASIC = _GPU_FIELDS_FULL[:7]

_CSV_FIELDS = (
    "num_envs",
    "status",
    "selected",
    "device",
    "startup_s",
    "collect_s",
    "learn_s",
    "iteration_s",
    "samples_per_s",
    "updates_per_hour",
    "batch_size",
    "minibatch_size",
    "num_steps_per_env",
    "num_mini_batches",
    "cpu_cores",
    "host_rss_gib",
    "torch_peak_gib",
    "gpu_sm_util_collect_pct",
    "gpu_sm_util_learn_pct",
    "gpu_mem_bandwidth_util_collect_pct",
    "gpu_mem_bandwidth_util_learn_pct",
    "gpu_sm_clock_collect_mhz",
    "gpu_sm_clock_learn_mhz",
    "gpu_sm_clock_frac_collect",
    "gpu_sm_clock_frac_learn",
    "gpu_power_collect_w",
    "gpu_power_learn_w",
    "gpu_mem_used_peak_mib",
    "gpu_throttle_fraction",
    "gpu_samples_collect",
    "gpu_samples_learn",
    "overflow_warnings",
    "gpu_monitor",
    "error",
)

_MISSING = {"", "[N/A]", "[Not Supported]", "N/A"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4096],
        help="Environment counts, probed from smallest to largest.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="Measured learning iterations after warmup.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default: cuda:0 when CUDA is available, otherwise cpu.",
    )
    parser.add_argument(
        "--gpu-sample-ms",
        type=int,
        default=100,
        help="nvidia-smi sampling period in milliseconds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/benchmarks/ladder_envs.csv"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-envs", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def device_index(device: str) -> int | None:
    """Return the CUDA index nvidia-smi should sample, or None on CPU."""

    if not device.startswith("cuda"):
        return None
    if ":" not in device:
        return 0
    return int(device.split(":", 1)[1])


def parse_gpu_line(line: str, fields: Sequence[str]) -> dict[str, Any] | None:
    """Parse one ``nvidia-smi --format=csv,noheader,nounits`` sample."""

    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) != len(fields):
        return None
    parsed: dict[str, Any] = {}
    for name, raw in zip(fields, parts, strict=True):
        if raw in _MISSING:
            parsed[name] = None
        elif raw in {"Active", "Not Active"}:
            parsed[name] = raw == "Active"
        else:
            try:
                parsed[name] = float(raw)
            except ValueError:
                return None
    return parsed


def rss_gib(ru_maxrss: int, platform: str) -> float:
    """Convert ``resource.ru_maxrss`` to gibibytes.

    macOS reports bytes. Linux reports kilobytes.
    """

    if platform == "darwin":
        return ru_maxrss / (1024**3)
    return ru_maxrss / (1024**2)


def classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if "outofmemory" in text or "out of memory" in text:
        return "oom"
    if "nan" in text:
        return "nan"
    return "failed"


def validate_batch(num_envs: int, num_steps_per_env: int, num_mini_batches: int) -> None:
    if num_envs <= 0 or num_steps_per_env <= 0 or num_mini_batches <= 0:
        raise ValueError("num_envs, num_steps_per_env, and num_mini_batches must be positive")
    batch = num_envs * num_steps_per_env
    if batch % num_mini_batches != 0:
        raise ValueError(
            f"num_mini_batches={num_mini_batches} must divide "
            f"num_envs * num_steps_per_env ({batch})"
        )


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _samples_in_windows(
    samples: Sequence[tuple[float, Mapping[str, Any]]],
    windows: Sequence[tuple[float, float]],
) -> list[Mapping[str, Any]]:
    selected = []
    for timestamp, sample in samples:
        if any(start <= timestamp < end for start, end in windows):
            selected.append(sample)
    return selected


def _clock_fraction(samples: Sequence[Mapping[str, Any]]) -> float | None:
    fractions = []
    for sample in samples:
        clock = sample.get("sm_clock_mhz")
        maximum = sample.get("sm_clock_max_mhz")
        if isinstance(clock, (int, float)) and isinstance(maximum, (int, float)) and maximum > 0:
            fractions.append(float(clock) / float(maximum))
    return _median(fractions)


def _throttle_fraction(samples: Sequence[Mapping[str, Any]]) -> float | None:
    known = 0
    active = 0
    for sample in samples:
        flags = [
            flag
            for flag in (sample.get("sw_power_cap"), sample.get("hw_thermal"))
            if isinstance(flag, bool)
        ]
        if not flags:
            continue
        known += 1
        if any(flags):
            active += 1
    if known == 0:
        return None
    return active / known


def summarize_run(
    *,
    num_envs: int,
    device: str,
    num_steps_per_env: int,
    num_mini_batches: int,
    startup_s: float,
    warmup: int,
    iterations: Sequence[Mapping[str, Any]],
    gpu_samples: Sequence[tuple[float, Mapping[str, Any]]],
    cpu_cores: float | None,
    host_rss_gib: float | None,
    torch_peak_bytes: int | None,
    overflow_warnings: int,
    gpu_monitor: str,
) -> dict[str, Any]:
    """Reduce one completed size to steady-state throughput and device load."""

    measured = list(iterations[warmup:])
    if not measured:
        raise RuntimeError(f"expected measured iterations after warmup={warmup}")
    collect_s = _median([float(row["collect_s"]) for row in measured])
    learn_s = _median([float(row["learn_s"]) for row in measured])
    iteration_s = _median(
        [float(row["collect_s"]) + float(row["learn_s"]) for row in measured]
    )
    assert collect_s is not None and learn_s is not None and iteration_s is not None
    collect_windows = [tuple(row["collect_window"]) for row in measured]
    learn_windows = [tuple(row["learn_window"]) for row in measured]
    collect_samples = _samples_in_windows(gpu_samples, collect_windows)
    learn_samples = _samples_in_windows(gpu_samples, learn_windows)
    phase_samples = collect_samples + learn_samples
    batch_size = num_envs * num_steps_per_env
    return {
        "num_envs": num_envs,
        "status": "ok",
        "device": device,
        "startup_s": round(startup_s, 3),
        "collect_s": round(collect_s, 4),
        "learn_s": round(learn_s, 4),
        "iteration_s": round(iteration_s, 4),
        "samples_per_s": round(batch_size / iteration_s, 1),
        "updates_per_hour": round(3600.0 / iteration_s, 1),
        "batch_size": batch_size,
        "minibatch_size": batch_size // num_mini_batches,
        "num_steps_per_env": num_steps_per_env,
        "num_mini_batches": num_mini_batches,
        "cpu_cores": None if cpu_cores is None else round(cpu_cores, 2),
        "host_rss_gib": None if host_rss_gib is None else round(host_rss_gib, 3),
        "torch_peak_gib": (
            None if not torch_peak_bytes else round(torch_peak_bytes / (1024**3), 3)
        ),
        "gpu_sm_util_collect_pct": _round_optional(
            _median(_numbers(collect_samples, "sm_util"))
        ),
        "gpu_sm_util_learn_pct": _round_optional(
            _median(_numbers(learn_samples, "sm_util"))
        ),
        "gpu_mem_bandwidth_util_collect_pct": _round_optional(
            _median(_numbers(collect_samples, "mem_util"))
        ),
        "gpu_mem_bandwidth_util_learn_pct": _round_optional(
            _median(_numbers(learn_samples, "mem_util"))
        ),
        "gpu_sm_clock_collect_mhz": _round_optional(
            _median(_numbers(collect_samples, "sm_clock_mhz")),
            0,
        ),
        "gpu_sm_clock_learn_mhz": _round_optional(
            _median(_numbers(learn_samples, "sm_clock_mhz")),
            0,
        ),
        "gpu_sm_clock_frac_collect": _round_optional(_clock_fraction(collect_samples), 3),
        "gpu_sm_clock_frac_learn": _round_optional(_clock_fraction(learn_samples), 3),
        "gpu_power_collect_w": _round_optional(_median(_numbers(collect_samples, "power_w"))),
        "gpu_power_learn_w": _round_optional(_median(_numbers(learn_samples, "power_w"))),
        "gpu_mem_used_peak_mib": _round_optional(
            _peak(_numbers(phase_samples, "mem_used_mib")),
            0,
        ),
        "gpu_throttle_fraction": _round_optional(_throttle_fraction(phase_samples), 3),
        "gpu_samples_collect": len(collect_samples),
        "gpu_samples_learn": len(learn_samples),
        "overflow_warnings": overflow_warnings,
        "gpu_monitor": gpu_monitor,
    }


def _numbers(samples: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [
        float(sample[key])
        for sample in samples
        if isinstance(sample.get(key), (int, float))
    ]


def _peak(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return max(values)


def _round_optional(value: float | None, digits: int = 1) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def choose_num_envs(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Pick the smallest count whose samples/s stay within 10% of the peak."""

    ok = [
        row
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("samples_per_s"), (int, float))
    ]
    if not ok:
        return None
    peak = max(float(row["samples_per_s"]) for row in ok)
    near = [row for row in ok if float(row["samples_per_s"]) >= 0.9 * peak]
    return int(min(near, key=lambda row: int(row["num_envs"]))["num_envs"])


class _OverflowTee:
    """Forward a text stream and keep lines that look like constraint overflows."""

    def __init__(self, underlying: Any, matches: list[str]) -> None:
        self._underlying = underlying
        self._matches = matches
        self._pending = ""

    def write(self, data: str) -> int:
        self._underlying.write(data)
        self._pending += data
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if _OVERFLOW_RE.search(line):
                self._matches.append(line)
        return len(data)

    def flush(self) -> None:
        self._underlying.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


class GpuMonitor:
    """Sample one GPU with a long-running ``nvidia-smi -lms`` process."""

    def __init__(self, device_index: int | None, sample_ms: int) -> None:
        self.device_index = device_index
        self.sample_ms = sample_ms
        self.samples: list[tuple[float, dict[str, Any]]] = []
        self.fields: tuple[str, ...] = ()
        self.error = "cpu" if device_index is None else "unavailable"
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device_index is None:
            return
        queries = (
            (_GPU_QUERY_FULL, _GPU_FIELDS_FULL),
            (_GPU_QUERY_BASIC, _GPU_FIELDS_BASIC),
        )
        last_error = "nvidia-smi exited"
        for query, fields in queries:
            try:
                proc = subprocess.Popen(
                    [
                        "nvidia-smi",
                        f"--id={self.device_index}",
                        f"--query-gpu={','.join(query)}",
                        "--format=csv,noheader,nounits",
                        "-lms",
                        str(self.sample_ms),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
            except FileNotFoundError:
                self.error = "nvidia-smi not found"
                return
            time.sleep(0.4)
            if proc.poll() is None:
                self._proc = proc
                self.fields = fields
                self.error = "ok"
                self._thread = threading.Thread(target=self._read, args=(proc,), daemon=True)
                self._thread.start()
                return
            last_error = "nvidia-smi exited"
            proc.kill()
        self.error = last_error

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _read(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            parsed = parse_gpu_line(line, self.fields)
            if parsed is not None:
                self.samples.append((time.perf_counter(), parsed))


def _run_worker(num_envs: int, args: argparse.Namespace) -> dict[str, Any]:
    import resource

    import torch

    from train_mimic.app import (
        build_runner_cfg_dict,
        import_training_stack,
        load_task_components,
    )
    from train_mimic.scripts.train import _resolve_device
    from train_mimic.tasks.tracking.config.constants import LADDER_RL_TASK
    from train_mimic.tasks.tracking.config.env import (
        make_g1_ladder_training_robot_cfg,
        resolve_g1_training_xml,
    )
    from train_mimic.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

    device = _resolve_device(args, torch)
    (
        _torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    configure_torch_backends()
    _task, env_cfg, agent_cfg, runner_cls = load_task_components(
        LADDER_RL_TASK,
        load_env_cfg=load_env_cfg,
        load_rl_cfg=load_rl_cfg,
        load_runner_cls=load_runner_cls,
    )
    steps = int(agent_cfg.num_steps_per_env)
    minibatches = int(agent_cfg.algorithm.num_mini_batches)
    validate_batch(num_envs, steps, minibatches)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args.seed
    robot_xml = resolve_g1_training_xml(None)
    env_cfg.scene.entities["robot"] = make_g1_ladder_training_robot_cfg(robot_xml)
    env_cfg.robot_xml = str(robot_xml)
    total = args.warmup + args.iterations
    agent_cfg.max_iterations = total
    agent_cfg.save_interval = 10**9
    agent_cfg.experiment_name = "g1_ladder_rl_bench"

    iterations: list[dict[str, Any]] = []
    cpu_marks: dict[str, tuple[float, float]] = {}
    original_log = MotionTrackingOnPolicyRunner._log_one_based_iteration

    def _capture(runner: Any, **kwargs: Any) -> None:
        now = time.perf_counter()
        collect_s = float(kwargs["collect_time"])
        learn_s = float(kwargs["learn_time"])
        learn_start = now - learn_s
        iterations.append(
            {
                "collect_s": collect_s,
                "learn_s": learn_s,
                "collect_window": (learn_start - collect_s, learn_start),
                "learn_window": (learn_start, now),
            }
        )
        completed = len(iterations)
        if completed == args.warmup:
            cpu_marks["start"] = (now, time.process_time())
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
        if completed == total:
            cpu_marks["end"] = (time.perf_counter(), time.process_time())
        return original_log(runner, **kwargs)

    monitor = GpuMonitor(device_index(device), args.gpu_sample_ms)
    overflow: list[str] = []
    stdout_tee = _OverflowTee(sys.stdout, overflow)
    stderr_tee = _OverflowTee(sys.stderr, overflow)
    previous_stdout, previous_stderr = sys.stdout, sys.stderr
    env = None
    try:
        monitor.start()
        sys.stdout = stdout_tee
        sys.stderr = stderr_tee
        started = time.perf_counter()
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        with tempfile.TemporaryDirectory(prefix="ladder-envs-") as log_dir:
            runner = (runner_cls or MjlabOnPolicyRunner)(
                env,
                build_runner_cfg_dict(agent_cfg),
                log_dir=log_dir,
                device=device,
            )
            runner.save = lambda *_args, **_kwargs: None
            startup_s = time.perf_counter() - started
            print(
                f"[INFO] num_envs={num_envs} device={device} startup_s={startup_s:.1f} "
                f"running {total} iterations",
                flush=True,
            )
            MotionTrackingOnPolicyRunner._log_one_based_iteration = _capture
            if args.warmup == 0:
                cpu_marks["start"] = (time.perf_counter(), time.process_time())
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
            runner.learn(num_learning_iterations=total, init_at_random_ep_len=False)
        cpu_cores = None
        if "start" in cpu_marks and "end" in cpu_marks:
            wall = cpu_marks["end"][0] - cpu_marks["start"][0]
            if wall > 0:
                cpu_cores = (cpu_marks["end"][1] - cpu_marks["start"][1]) / wall
        torch_peak = (
            int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0
        )
        monitor.stop()
        return summarize_run(
            num_envs=num_envs,
            device=device,
            num_steps_per_env=steps,
            num_mini_batches=minibatches,
            startup_s=startup_s,
            warmup=args.warmup,
            iterations=iterations,
            gpu_samples=list(monitor.samples),
            cpu_cores=cpu_cores,
            host_rss_gib=rss_gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform),
            torch_peak_bytes=torch_peak,
            overflow_warnings=len(overflow),
            gpu_monitor=monitor.error,
        )
    finally:
        MotionTrackingOnPolicyRunner._log_one_based_iteration = original_log
        sys.stdout = previous_stdout
        sys.stderr = previous_stderr
        monitor.stop()
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()


def _probe(num_envs: int, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--num-envs",
        str(num_envs),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--gpu-sample-ms",
        str(args.gpu_sample_ms),
    ]
    if args.device is not None:
        command.extend(["--device", args.device])
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result = None
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            if line.startswith("BENCH "):
                result = json.loads(line[len("BENCH ") :])
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise
    if result is not None:
        return result
    status = "oom" if proc.returncode in (-9, 137) else "failed"
    return {
        "num_envs": num_envs,
        "status": status,
        "error": f"worker exited {proc.returncode} without a result",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key, "")
                    for key in _CSV_FIELDS
                }
            )


def _print_row(row: Mapping[str, Any]) -> None:
    if row.get("status") != "ok":
        print(
            f"[INFO] num_envs={row.get('num_envs')} status={row.get('status')} "
            f"{row.get('error', '')}",
            flush=True,
        )
        return
    print(
        "[INFO] "
        f"num_envs={row['num_envs']} samples/s={row['samples_per_s']} "
        f"updates/h={row['updates_per_hour']} "
        f"iter={row['iteration_s']}s collect={row['collect_s']}s learn={row['learn_s']}s "
        f"batch={row['batch_size']} minibatch={row['minibatch_size']} "
        f"sm_collect={row['gpu_sm_util_collect_pct']}% "
        f"sm_learn={row['gpu_sm_util_learn_pct']}% "
        f"mem_bw_collect={row['gpu_mem_bandwidth_util_collect_pct']}% "
        f"torch_peak={row['torch_peak_gib']}GiB "
        f"cpu_cores={row['cpu_cores']} overflow={row['overflow_warnings']}",
        flush=True,
    )


def _print_choice(rows: Sequence[Mapping[str, Any]], chosen: int | None) -> None:
    if chosen is None:
        print("[INFO] no size completed", flush=True)
        return
    ok = [row for row in rows if row.get("status") == "ok"]
    peak = max(ok, key=lambda row: float(row["samples_per_s"]))
    print(f"[INFO] use --num_envs {chosen}", flush=True)
    if int(peak["num_envs"]) != chosen:
        print(
            f"[INFO] {peak['num_envs']} envs is the samples/s peak; "
            f"{chosen} stays within 10% at a smaller PPO batch",
            flush=True,
        )
    failed = next((row for row in rows if row.get("status") != "ok"), None)
    if failed is not None:
        print(
            f"[INFO] stopped at num_envs={failed['num_envs']} ({failed['status']})",
            flush=True,
        )
    elif int(peak["num_envs"]) == max(int(row["num_envs"]) for row in ok):
        print("[INFO] the largest size was fastest; try a larger --sizes value", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be >= 0 and --iterations must be positive")
    if args.gpu_sample_ms <= 0:
        raise ValueError("--gpu-sample-ms must be positive")
    if args.worker:
        if args.num_envs is None or args.num_envs <= 0:
            raise ValueError("--num-envs must be positive in the worker")
        try:
            row = _run_worker(args.num_envs, args)
        except Exception as exc:
            traceback.print_exc()
            row = {
                "num_envs": args.num_envs,
                "status": classify_failure(exc),
                "error": f"{type(exc).__name__}: {exc}",
            }
        print("BENCH " + json.dumps(row), flush=True)
        return

    sizes = sorted({int(size) for size in args.sizes})
    if any(size <= 0 for size in sizes):
        raise ValueError("--sizes must be positive")
    rows: list[dict[str, Any]] = []
    for size in sizes:
        print(f"[INFO] probing num_envs={size}", flush=True)
        row = _probe(size, args)
        rows.append(row)
        _print_row(row)
        if row.get("status") != "ok":
            break
    chosen = choose_num_envs(rows)
    for row in rows:
        row["selected"] = int(row.get("num_envs") == chosen and row.get("status") == "ok")
    _write_csv(args.out, rows)
    print(f"[INFO] wrote {args.out}", flush=True)
    _print_choice(rows, chosen)


if __name__ == "__main__":
    main()
