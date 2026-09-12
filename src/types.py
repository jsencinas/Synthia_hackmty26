from dataclasses import dataclass, field


@dataclass
class StageResult:
    stage: str
    is_synthetic: bool
    p_synthetic: float
    certainty_pct: float
    features: dict = field(default_factory=dict)
    error: str | None = None


def certainty_from_probability(p_synthetic: float) -> float:
    return max(p_synthetic, 1.0 - p_synthetic) * 100.0


def label_from_probability(p_synthetic: float) -> bool:
    return p_synthetic >= 0.5
