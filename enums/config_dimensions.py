from enum import Enum
from dataclasses import dataclass
import re


class GpuType(str, Enum):
    A100 = "a100"
    H100 = "h100"
    B200 = "b200"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    SGD = "sgd"


class ModelFamily(str, Enum):
    OPT = "opt"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"


MODEL_FAMILY_MAP: dict[str, dict[str, dict[int, str]]] = {
    ModelSource.HUGGINGFACE: {
        ModelFamily.OPT: {
            125: "facebook/opt-125m",
            350: "facebook/opt-350m",
            1300: "facebook/opt-1.3b",
            2700: "facebook/opt-2.7b",
        },
        ModelFamily.QWEN: {
            500: "Qwen/Qwen2.5-0.5B",
            1500: "Qwen/Qwen2.5-1.5B",
            3000: "Qwen/Qwen2.5-3B",
        },
        ModelFamily.DEEPSEEK: {
            1300: "deepseek-ai/deepseek-llm-1.3b",
            7000: "deepseek-ai/deepseek-llm-7b",
        },
    },
    ModelSource.MODELSCOPE: {
        ModelFamily.QWEN: {
            500: "qwen/Qwen2.5-0.5B",
            1500: "qwen/Qwen2.5-1.5B",
            3000: "qwen/Qwen2.5-3B",
        },
        ModelFamily.DEEPSEEK: {
            1300: "deepseek-ai/deepseek-llm-1.3b",
            7000: "deepseek-ai/deepseek-llm-7b",
        },
    },
}

DATASET_MAP: dict[str, dict[str, str]] = {
    ModelSource.HUGGINGFACE: {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
    },
    ModelSource.MODELSCOPE: {
        "path": "modelscope/wikitext",
        "name": "wikitext-2-raw-v1",
    },
}


@dataclass
class LlmConfig:
    batch_size: int
    sequence_length: int
    model_params_millions: int
    gradient_accumulation: int
    precision: Precision
    power_cap_percent: int
    optimizer: OptimizerType
    num_gpus: int
    model_family: ModelFamily = ModelFamily.OPT

    BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    SEQUENCE_LENGTHS = [256, 512, 1024, 2048, 4096]
    MODEL_PARAMS_MILLIONS = [125, 350, 1300, 2700]
    GRADIENT_ACCUMULATIONS = [1, 2, 4, 8, 16]
    PRECISIONS = [Precision.FP32, Precision.FP16, Precision.BF16]
    POWER_CAPS = [60, 70, 80, 90, 100]
    OPTIMIZERS = [OptimizerType.ADAMW, OptimizerType.SGD]
    NUM_GPUS = [1, 2, 4, 8]

    def to_filename(self, sequence_number: int = 1) -> str:
        return (
            f"bs{self.batch_size:02d}"
            f"_sl{self.sequence_length:04d}"
            f"_m{self.model_params_millions:05d}"
            f"_ga{self.gradient_accumulation:02d}"
            f"_p{self.precision.value}"
            f"_pc{self.power_cap_percent:03d}"
            f"_{self.optimizer.value}"
            f"_ng{self.num_gpus}"
            f"_{self.model_family.value}"
            f"_{sequence_number:03d}"
        )

    @staticmethod
    def from_filename(name: str) -> "LlmConfig":
        pattern = (
            r"bs(\d+)_sl(\d+)_m(\d+)_ga(\d+)_p(fp\d+|bf\d+)_"
            r"pc(\d+)_(adamw|sgd)_ng(\d+)_(opt|qwen|deepseek)_\d{3}"
        )
        match = re.match(pattern, name)
        if not match:
            raise ValueError(f"Filename does not match schema: {name}")

        return LlmConfig(
            batch_size=int(match.group(1)),
            sequence_length=int(match.group(2)),
            model_params_millions=int(match.group(3)),
            gradient_accumulation=int(match.group(4)),
            precision=Precision(match.group(5)),
            power_cap_percent=int(match.group(6)),
            optimizer=OptimizerType(match.group(7)),
            num_gpus=int(match.group(8)),
            model_family=ModelFamily(match.group(9)),
        )

    def to_input_vector(self) -> list:
        precision_map = {Precision.FP32: 32, Precision.FP16: 16, Precision.BF16: 16}
        optimizer_map = {OptimizerType.ADAMW: 0, OptimizerType.SGD: 1}
        family_map = {ModelFamily.OPT: 0, ModelFamily.QWEN: 1, ModelFamily.DEEPSEEK: 2}
        return [
            self.batch_size,
            self.sequence_length,
            self.model_params_millions,
            self.gradient_accumulation,
            precision_map[self.precision],
            self.power_cap_percent,
            optimizer_map[self.optimizer],
            self.num_gpus,
            family_map[self.model_family],
        ]

    @property
    def input_vector_labels(self) -> list[str]:
        return [
            "batch_size",
            "sequence_length",
            "model_params_M",
            "gradient_accumulation",
            "precision_bits",
            "power_cap_percent",
            "optimizer",
            "num_gpus",
            "model_family",
        ]

    def full_csv_path(self, gpu_type: GpuType, sequence_number: int = 1) -> str:
        return f"llm/{gpu_type.value}/{self.to_filename(sequence_number)}.csv"
