import subprocess
import time
import csv
import os
import psutil
from datetime import datetime

from enums.csv_columns import (
    GPU_NVIDIA_SMI_QUERY,
    GPU_COLUMNS_ORDERED,
    CPU_COLUMNS_ORDERED,
    MetaColumn,
)

SAMPLE_INTERVAL_MS = 20
SAMPLE_RATE_HZ = 1000 / SAMPLE_INTERVAL_MS


def get_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    return len(result.stdout.strip().split("\n"))


def collect_gpu_sample(gpu_index: int) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            f"--query-gpu={GPU_NVIDIA_SMI_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True
    )
    values = [v.strip() for v in result.stdout.strip().split(",")]
    return values[1:]


def collect_cpu_sample() -> list[str]:
    cpu_util = psutil.cpu_percent(percpu=False)
    cpu_freq = psutil.cpu_freq()
    cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
    return [str(cpu_util), str(cpu_freq_mhz)]


def get_power_limit_range(gpu_index: int) -> tuple[float, float]:
    """Returns (min_power_limit, max_power_limit) in watts."""
    result_min = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=power.min_limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True
    )
    result_max = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=power.max_limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True
    )
    min_limit = float(result_min.stdout.strip())
    max_limit = float(result_max.stdout.strip())
    return min_limit, max_limit


def set_power_cap(gpu_indices: list[int], power_limit_watts: int):
    min_limit, max_limit = get_power_limit_range(gpu_indices[0])
    clamped_watts = max(min_limit, min(max_limit, power_limit_watts))
    
    for gpu_idx in gpu_indices:
        subprocess.run(
            ["sudo", "nvidia-smi", f"--id={gpu_idx}", f"-pl={clamped_watts}"],
            check=True
        )


def get_tdp_watts(gpu_index: int) -> float:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def build_column_headers(num_gpus: int) -> list[str]:
    headers = [MetaColumn.SAMPLE_INDEX.value, MetaColumn.TIMESTAMP.value]

    for gpu_idx in range(num_gpus):
        for col in GPU_COLUMNS_ORDERED:
            headers.append(f"gpu{gpu_idx}_{col.value}")

    for col in CPU_COLUMNS_ORDERED:
        headers.append(col.value)

    return headers


def wait_for_training_ready(ready_signal_path: str, timeout_seconds: int = 300) -> bool:
    start = time.monotonic()
    while not os.path.exists(ready_signal_path):
        if time.monotonic() - start > timeout_seconds:
            print(f"Training did not become ready within {timeout_seconds}s, aborting")
            return False
        time.sleep(1)
    print(f"Training ready after {time.monotonic() - start:.1f}s")
    return True


def record_session(
    output_csv_path: str,
    duration_seconds: int = 900,
    power_cap_watts: int = None,
    ready_signal_path: str = None,
):
    """Record power data from ALL GPUs on the node, regardless of session size."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    if ready_signal_path is not None:
        if not wait_for_training_ready(ready_signal_path):
            return 0

    num_gpus = get_gpu_count()
    all_gpu_indices = list(range(num_gpus))

    if power_cap_watts is not None:
        set_power_cap(all_gpu_indices, power_cap_watts)

    all_columns = build_column_headers(num_gpus)

    psutil.cpu_percent(percpu=True)

    total_samples = int(duration_seconds * SAMPLE_RATE_HZ)
    samples_collected = 0

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(all_columns)

        start_time = time.monotonic()

        for i in range(total_samples):
            target_time = start_time + (i * SAMPLE_INTERVAL_MS / 1000.0)

            timestamp = datetime.now().isoformat()
            row = [i, timestamp]

            for gpu_idx in all_gpu_indices:
                gpu_sample = collect_gpu_sample(gpu_idx)
                row.extend(gpu_sample)

            cpu_sample = collect_cpu_sample()
            row.extend(cpu_sample)

            writer.writerow(row)
            samples_collected += 1

            sleep_duration = target_time - time.monotonic()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            if samples_collected % 2500 == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"  All {num_gpus} GPUs | {samples_collected}/{total_samples} samples "
                    f"({elapsed:.0f}s elapsed)"
                )

    if power_cap_watts is not None:
        tdp = get_tdp_watts(0)
        set_power_cap(all_gpu_indices, int(tdp))

    return samples_collected
