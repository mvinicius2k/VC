from dataclasses import dataclass
@dataclass
class DetectParams:
    scale: float
    neightboor: float
    min_size: tuple[int,int]
    max_size: tuple[int,int]