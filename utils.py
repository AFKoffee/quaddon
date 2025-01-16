import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class WeightDistribution:
    min: list[float]
    q1: list[float]
    q25: list[float]
    q75: list[float]
    q99: list[float]
    max: list[float]

    def find_abs_max(self) -> float:
        min_max = np.max(np.abs(self.min))
        max_max = np.max(self.max)
        return max(min_max, max_max)
    
    def get_nfeatures(self) -> int:
        assert len(self.min) == len(self.q1)
        assert len(self.q1) == len(self.q25)
        assert len(self.q25) == len(self.q75)
        assert len(self.q75) == len(self.q99)
        assert len(self.q99) == len(self.max)

        return len(self.min)
    

@dataclass
class Results:
    incoherence: dict[str, list[float]] = field(default_factory=lambda: defaultdict(lambda: []))
    mae: dict[str, list[float]] = field(default_factory=lambda: defaultdict(lambda: []))

    def push_incoherence(self, inc_orig: float, inc_had: float):
        self.incoherence["original"].append(inc_orig)
        self.incoherence["hadamard"].append(inc_had)
    
    def push_mae(self, mae_orig: float, mae_had: float):
        self.mae["original"].append(mae_orig)
        self.mae["hadamard"].append(mae_had)

    def get_orig_incoherence(self) -> list[float]:
        return self.incoherence["original"]
    
    def get_had_incoherence(self) -> list[float]:
        return self.incoherence["hadamard"]
    
    def get_orig_mae(self) -> list[float]:
        return self.mae["original"]
    
    def get_had_mae(self) -> list[float]:
        return self.mae["hadamard"]