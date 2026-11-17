from enum import Enum


class GpuColumn(str, Enum):
    POWER_DRAW_W = "power_draw_W"
    TEMPERATURE_GPU_C = "temperature_gpu_C"
    UTILIZATION_GPU_PCT = "utilization_gpu_pct"
    UTILIZATION_MEMORY_PCT = "utilization_memory_pct"
    CLOCKS_SM_MHZ = "clocks_sm_MHz"
    CLOCKS_MEMORY_MHZ = "clocks_memory_MHz"
    PSTATE = "pstate"
    MEMORY_USED_MIB = "memory_used_MiB"
    MEMORY_FREE_MIB = "memory_free_MiB"
    MEMORY_TOTAL_MIB = "memory_total_MiB"
    POWER_LIMIT_W = "power_limit_W"


class CpuColumn(str, Enum):
    UTILIZATION_PCT = "cpu_utilization_pct"
    FREQ_MHZ = "cpu_freq_MHz"


class MetaColumn(str, Enum):
    SAMPLE_INDEX = "sample_index"
    TIMESTAMP = "timestamp"


GPU_NVIDIA_SMI_QUERY = (
    "timestamp,"
    "power.draw,"
    "temperature.gpu,"
    "utilization.gpu,"
    "utilization.memory,"
    "clocks.current.sm,"
    "clocks.current.memory,"
    "pstate,"
    "memory.used,"
    "memory.free,"
    "memory.total,"
    "power.limit"
)

GPU_COLUMNS_ORDERED = [
    GpuColumn.POWER_DRAW_W,
    GpuColumn.TEMPERATURE_GPU_C,
    GpuColumn.UTILIZATION_GPU_PCT,
    GpuColumn.UTILIZATION_MEMORY_PCT,
    GpuColumn.CLOCKS_SM_MHZ,
    GpuColumn.CLOCKS_MEMORY_MHZ,
    GpuColumn.PSTATE,
    GpuColumn.MEMORY_USED_MIB,
    GpuColumn.MEMORY_FREE_MIB,
    GpuColumn.MEMORY_TOTAL_MIB,
    GpuColumn.POWER_LIMIT_W,
]

CPU_COLUMNS_ORDERED = [
    CpuColumn.UTILIZATION_PCT,
    CpuColumn.FREQ_MHZ,
]
