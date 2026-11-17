import random as _random

from enums.config_dimensions import LlmConfig


def _lhs_unit_samples(n: int, d: int, seed: int) -> list[list[float]]:
    rng = _random.Random(seed)
    samples = []
    for i in range(n):
        point = [(rng.random() + j) / n for j in range(d)]
        samples.append(point)
    for dim in range(d):
        col = [samples[i][dim] for i in range(n)]
        rng.shuffle(col)
        for i in range(n):
            samples[i][dim] = col[i]
    return samples


def _normalize(configs: list[LlmConfig]) -> list[list[float]]:
    if not configs:
        return []
    vectors = [c.to_input_vector() for c in configs]
    d = len(vectors[0])
    mins = [min(v[j] for v in vectors) for j in range(d)]
    maxs = [max(v[j] for v in vectors) for j in range(d)]
    ranges = [maxs[j] - mins[j] if maxs[j] != mins[j] else 1.0 for j in range(d)]
    return [[(v[j] - mins[j]) / ranges[j] for j in range(d)] for v in vectors]


def sample_lhs(feasible: list[LlmConfig], n: int, seed: int) -> list[LlmConfig]:
    d = len(feasible[0].to_input_vector())
    lhs_points = _lhs_unit_samples(n, d, seed)
    norm_configs = _normalize(feasible)

    selected: list[LlmConfig] = []
    used: set[int] = set()

    for point in lhs_points:
        best_idx, best_dist = -1, float("inf")
        for i, nc in enumerate(norm_configs):
            if i in used:
                continue
            dist = sum((a - b) ** 2 for a, b in zip(point, nc))
            if dist < best_dist:
                best_dist, best_idx = dist, i
        if best_idx >= 0:
            selected.append(feasible[best_idx])
            used.add(best_idx)

    return selected
