import numpy as np
from dataclasses import dataclass, field, is_dataclass, asdict
from collections import defaultdict
import json

@dataclass
class Config:
    bitwidth: int
    display: bool
    save: bool
    quantile_analysis: bool
    validate: bool
    out_dir: str
    module: str
    seed: int


@dataclass
class WeightDistribution:
    layer_idx: int
    is_transformed: bool
    min: list[float] = field(default_factory=lambda: [])
    p1: list[float] = field(default_factory=lambda: [])
    p25: list[float] = field(default_factory=lambda: [])
    p75: list[float] = field(default_factory=lambda: [])
    p99: list[float] = field(default_factory=lambda: [])
    max: list[float] = field(default_factory=lambda: [])

    def fill(
        self, 
        min: list[float], 
        p1: list[float], 
        p25: list[float],
        p75: list[float],
        p99: list[float],
        max: list[float]
    ):
        self.min = min
        self.p1 = p1
        self.p25 = p25
        self.p75 = p75
        self.p99 = p99
        self.max = max
        

    def find_abs_max(self) -> float:
        min_max = np.max(np.abs(self.min))
        max_max = np.max(self.max)
        return max(min_max, max_max)
    
    def get_nfeatures(self, validate: bool) -> int:
        if validate:
            assert len(self.min) == len(self.p1)
            assert len(self.p1) == len(self.p25)
            assert len(self.p25) == len(self.p75)
            assert len(self.p75) == len(self.p99)
            assert len(self.p99) == len(self.max)

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

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if is_dataclass(o) and not isinstance(o, type):
                return asdict(o)
            return super().default(o)