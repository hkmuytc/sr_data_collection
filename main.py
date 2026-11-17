import argparse
import json
import multiprocessing
import os
import subprocess
import tempfile
import time
from pathlib import Path

from enums.config_dimensions import GpuType, LlmConfig, ModelFamily, ModelSource, MODEL_FAMILY_MAP, OptimizerType, Precision
from sampling import sample_lhs
from workers.monitor import record_session, get_tdp_watts

SCRIPT_DIR = Path(__file__).parent
WORKERS_DIR = SCRIPT_DIR / "workers"


def main() -> None:
    args = parse_cli_arguments()

    model_family = ModelFamily(args.model_family)
    model_source = ModelSource(args.model_source)
    campaigns_to_be_executed = generate_campaign_configs(
        args.num_samples, args.seed, args.gpu_memory, model_family, model_source
    )
    gpu_type = GpuType(args.gpu)

    print(f"Campaign: {len(campaigns_to_be_executed)} sessions on {gpu_type.value} ({model_family.value} via {model_source.value})")

    for idx, config in enumerate(campaigns_to_be_executed, start=1):
        run_training_session(idx, config, gpu_type, args.output_dir, args.duration, model_family, model_source)


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=["a100", "h100", "b200"], required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--gpu-memory", type=int, default=80, help="Per-GPU VRAM in GB (default: 80)")
    parser.add_argument("--model-family", choices=["opt", "qwen", "deepseek"], default="opt")
    parser.add_argument("--model-source", choices=["huggingface", "modelscope"], default="huggingface")
    return parser.parse_args()


def run_training_session(
        sequence_number: int, config: LlmConfig,
        gpu_type: GpuType, output_dir: str, duration_seconds: int,
        model_family: ModelFamily, model_source: ModelSource
) -> None:
    csv_path = build_csv_path(config, sequence_number, gpu_type, output_dir)

    if csv_already_exists(csv_path, sequence_number):
        return

    ready_signal_path = f"/tmp/ready_{sequence_number}_{os.getpid()}.signal"
    config_file_path = write_training_config(config, model_family, model_source, ready_signal_path)

    monitor_process = start_power_monitor_to_wait_for_ready_signal(csv_path, duration_seconds, calculate_power_cap_watts(config), ready_signal_path)
    training_result = launch_training(config, config_file_path)

    stop_monitor_and_cleanup(monitor_process, config_file_path, ready_signal_path, sequence_number, csv_path, training_result)


def build_csv_path(config: LlmConfig, sequence_number: int, gpu_type: GpuType, output_dir: str) -> str:
    filename = config.to_filename(sequence_number)
    csv_dir = os.path.join(output_dir, "llm", gpu_type.value)
    os.makedirs(csv_dir, exist_ok=True)
    return os.path.join(csv_dir, f"{filename}.csv")


def csv_already_exists(csv_path: str, sequence_number: int) -> bool:
    if os.path.exists(csv_path):
        print(f"[{sequence_number}] Already exists, skipping: {csv_path}")
        return True
    return False


def calculate_power_cap_watts(config: LlmConfig) -> int:
    tdp = get_tdp_watts(0)
    power_cap_watts = int(tdp * config.power_cap_percent / 100)
    print(f"  GPUs: {config.num_gpus} | power cap: {power_cap_watts}W ({config.power_cap_percent}% of {tdp}W TDP)")
    return power_cap_watts


def write_training_config(config: LlmConfig, model_family: ModelFamily, model_source: ModelSource, ready_signal_path: str) -> str:
    config_dict = {
        "batch_size": config.batch_size,
        "sequence_length": config.sequence_length,
        "model_params_M": config.model_params_millions,
        "gradient_accumulation": config.gradient_accumulation,
        "precision": config.precision.value,
        "power_cap_percent": config.power_cap_percent,
        "optimizer": config.optimizer.value,
        "num_gpus": config.num_gpus,
        "model_family": model_family.value,
        "model_source": model_source.value,
        "ready_signal_path": ready_signal_path,
    }
    tmp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp")
    json.dump(config_dict, tmp_config)
    tmp_config.close()
    return tmp_config.name


def start_power_monitor_to_wait_for_ready_signal(csv_path: str, duration_seconds: int, power_cap_watts: int, ready_signal_path: str) -> multiprocessing.Process:
    monitor_process = multiprocessing.Process(
        target=record_session,
        args=(csv_path, duration_seconds, power_cap_watts, ready_signal_path),
    )
    monitor_process.start()
    return monitor_process


def launch_training(config: LlmConfig, config_file_path: str) -> subprocess.CompletedProcess:
    training_cmd = build_training_command(config, config_file_path)
    cuda_visible = ",".join(str(i) for i in range(config.num_gpus))
    training_env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible}
    return subprocess.run(training_cmd, env=training_env)


def stop_monitor_and_cleanup(
    monitor_process: multiprocessing.Process, config_file_path: str,
    ready_signal_path: str, sequence_number: int, csv_path: str,
    training_result: subprocess.CompletedProcess,
) -> None:
    monitor_process.join(timeout=30)
    if monitor_process.is_alive():
        monitor_process.terminate()
        monitor_process.join(timeout=5)

    os.unlink(config_file_path)
    if os.path.exists(ready_signal_path):
        os.unlink(ready_signal_path)

    if training_result.returncode != 0:
        print(f"[{sequence_number}] FAILED (exit code {training_result.returncode}): {csv_path}")
    else:
        print(f"[{sequence_number}] Done: {csv_path}")


def build_training_command(config: LlmConfig, config_file_path: str) -> list[str]:
    script = str(WORKERS_DIR / "train_llm.py")
    if config.num_gpus == 1:
        return ["python3", script, config_file_path]
    return [
        "torchrun",
        f"--nproc_per_node={config.num_gpus}",
        script,
        config_file_path,
    ]


def generate_campaign_configs(num_samples: int, seed: int, gpu_memory_gb: int, model_family: ModelFamily, model_source: ModelSource) -> list[LlmConfig]:
    all_configs = generate_all_configs(model_family, model_source)
    feasible = [c for c in all_configs if is_feasible(c, gpu_memory_gb)]

    if num_samples < len(feasible):
        campaigns = sample_lhs(feasible, num_samples, seed)
    else:
        campaigns = list(feasible)

    print(f"Config space: {len(all_configs)} total, {len(feasible)} feasible (fit in {gpu_memory_gb}GB), {len(campaigns)} sampled (LHS)")
    return campaigns


def generate_all_configs(model_family: ModelFamily, model_source: ModelSource) -> list[LlmConfig]:
    available_sizes = list(MODEL_FAMILY_MAP[model_source][model_family].keys())
    configs = []
    for bs in LlmConfig.BATCH_SIZES:
        for sl in LlmConfig.SEQUENCE_LENGTHS:
            for m in available_sizes:
                for ga in LlmConfig.GRADIENT_ACCUMULATIONS:
                    for p in LlmConfig.PRECISIONS:
                        for pc in LlmConfig.POWER_CAPS:
                            for opt in LlmConfig.OPTIMIZERS:
                                for ng in LlmConfig.NUM_GPUS:
                                    configs.append(
                                        LlmConfig(
                                            batch_size=bs,
                                            sequence_length=sl,
                                            model_params_millions=m,
                                            gradient_accumulation=ga,
                                            precision=p,
                                            power_cap_percent=pc,
                                            optimizer=opt,
                                            num_gpus=ng,
                                            model_family=model_family,
                                        )
                                    )
    return configs


def is_feasible(config: LlmConfig, gpu_memory_gb: int = 80) -> bool:
    bytes_per_param = {Precision.FP32: 4, Precision.FP16: 2, Precision.BF16: 2}
    param_bytes = config.model_params_millions * 1e6 * bytes_per_param[config.precision]

    optimizer_multiplier = {OptimizerType.ADAMW: 3, OptimizerType.SGD: 1}
    total_model_memory = param_bytes * optimizer_multiplier[config.optimizer]

    ga_divisor = config.gradient_accumulation if config.gradient_accumulation > 1 else 1
    activation_bytes = (
        config.batch_size
        * config.sequence_length
        * config.model_params_millions
        * 1e6
        * 2
        // ga_divisor
    )

    per_gpu_memory = (total_model_memory + activation_bytes) / config.num_gpus
    return per_gpu_memory < gpu_memory_gb * 1e9 * 0.85


if __name__ == "__main__":
    main()
