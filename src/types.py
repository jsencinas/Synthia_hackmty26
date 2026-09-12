from dataclasses import dataclass, field


@dataclass
class StageResult:
    stage: str
    is_synthetic: bool
    p_synthetic: float
    certainty_pct: float
    features: dict = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        self.p_synthetic = float(self.p_synthetic)
        if not 0.0 <= self.p_synthetic <= 1.0:
            raise ValueError("p_synthetic must be between 0 and 1.")
        self.is_synthetic = label_from_probability(self.p_synthetic)
        self.certainty_pct = certainty_from_probability(self.p_synthetic)

    @classmethod
    def failed(cls, stage: str, error: str) -> "StageResult":
        return cls(
            stage=stage,
            is_synthetic=False,
            p_synthetic=0.5,
            certainty_pct=50.0,
            features={},
            error=error,
        )


def certainty_from_probability(p_synthetic: float) -> float:
    return max(p_synthetic, 1.0 - p_synthetic) * 100.0


def label_from_probability(p_synthetic: float) -> bool:
    return p_synthetic >= 0.5
