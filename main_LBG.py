import math
import numpy as np
from typing import Dict, List, Set, Tuple

def predict_lgb_probability(features: list) -> float:
    raw_score = 0.0
    raw_score += (((-1.004911 if features[0] <= 12.500000 else -0.880178) if features[1] <= 0.000000 else (-0.887576 if features[4] <= 0.325758 else -0.976408)) if features[1] <= 1.500000 else -1.004911)
    raw_score += (((-0.068304 if features[0] <= 12.500000 else 0.051110) if features[1] <= 0.000000 else (0.044388 if features[4] <= 0.325758 else -0.039805)) if features[1] <= 1.500000 else -0.068304)
    raw_score += (((0.073726 if features[4] <= 0.469669 else 0.037504) if features[1] <= 0.000000 else (0.041393 if features[4] <= 0.325758 else -0.038533)) if features[1] <= 1.500000 else (-0.067095 if features[0] <= 709.500000 else -0.067095))
    raw_score += (((-0.068403 if features[0] <= 12.500000 else 0.044493) if features[1] <= 0.000000 else (0.073763 if features[4] <= 0.314145 else -0.010619)) if features[1] <= 1.500000 else (-0.065986 if features[0] <= 687.000000 else (-0.065986 if features[0] <= 709.500000 else -0.065986)))
    raw_score += (((0.038594 if features[1] <= 0.000000 else 0.007695) if features[0] <= 713.000000 else (0.155540 if features[0] <= 714.500000 else 0.162667)) if features[1] <= 1.500000 else (-0.064965 if features[0] <= 702.000000 else -0.064965))
    raw_score += (((0.056989 if features[4] <= 0.314145 else 0.000468) if features[2] <= 24.500000 else (0.147932 if features[4] <= 0.469669 else 0.035308)) if features[1] <= 1.500000 else -0.064024)
    raw_score += (((0.034070 if features[1] <= 0.000000 else 0.006479) if features[0] <= 713.000000 else (0.137204 if features[0] <= 714.500000 else 0.145708)) if features[1] <= 1.500000 else (-0.063154 if features[0] <= 702.000000 else -0.063154))
    raw_score += (((0.018262 if features[0] <= 18.500000 else -0.073428) if features[2] <= 4.500000 else (0.082497 if features[4] <= 0.314145 else 0.021441)) if features[1] <= 1.500000 else (-0.062349 if features[0] <= 662.000000 else -0.062349))
    raw_score += (((-0.069926 if features[1] <= 0.000000 else -0.076401) if features[0] <= 9.500000 else (0.138434 if features[4] <= 0.236111 else 0.020368)) if features[1] <= 1.500000 else -0.061603)
    raw_score += (((0.055136 if features[4] <= 0.469669 else 0.023474) if features[1] <= 0.000000 else (0.030361 if features[4] <= 0.325758 else -0.038542)) if features[1] <= 1.500000 else -0.060909)
    raw_score += (((-0.069634 if features[1] <= 0.000000 else -0.075213) if features[0] <= 9.500000 else (0.123996 if features[4] <= 0.236111 else 0.018117)) if features[1] <= 1.500000 else ((-0.060265 if features[0] <= 631.000000 else -0.060265) if features[0] <= 702.000000 else -0.060265))
    raw_score += (((0.014395 if features[4] <= 0.456439 else -0.052461) if features[2] <= 24.500000 else (0.122763 if features[4] <= 0.469669 else 0.026150)) if features[1] <= 1.500000 else -0.059664)
    raw_score += (((0.018174 if features[0] <= 698.000000 else -0.076103) if features[0] <= 713.000000 else (0.118135 if features[0] <= 714.500000 else 0.124508)) if features[1] <= 1.500000 else (-0.059105 if features[0] <= 709.500000 else -0.059105))
    raw_score += (((0.011534 if features[0] <= 18.500000 else -0.073905) if features[2] <= 4.500000 else (0.066381 if features[4] <= 0.314145 else 0.014940)) if features[1] <= 1.500000 else -0.058582)
    raw_score += (((-0.031706 if features[0] <= 34.000000 else 0.027384) if features[1] <= 0.000000 else (0.023482 if features[4] <= 0.325758 else -0.039035)) if features[1] <= 1.500000 else (-0.058094 if features[0] <= 709.500000 else -0.058094))
    raw_score += (((-0.068542 if features[1] <= 0.000000 else -0.075022) if features[0] <= 9.500000 else (0.109816 if features[4] <= 0.236111 else 0.013366)) if features[1] <= 1.500000 else (-0.057637 if features[0] <= 646.000000 else -0.057637))
    raw_score += (((0.014603 if features[0] <= 698.000000 else -0.075332) if features[0] <= 713.000000 else (0.107264 if features[0] <= 714.500000 else 0.112466)) if features[1] <= 1.500000 else (-0.057209 if features[0] <= 597.500000 else -0.057209))
    raw_score += (((0.050398 if features[4] <= 0.449495 else 0.016649) if features[1] <= 0.000000 else (0.020213 if features[4] <= 0.325758 else -0.038582)) if features[1] <= 1.500000 else (-0.056808 if features[0] <= 653.500000 else (-0.056808 if features[0] <= 694.500000 else -0.056808)))
    raw_score += (((0.008256 if features[0] <= 18.500000 else -0.073074) if features[2] <= 4.500000 else (0.056812 if features[4] <= 0.314145 else 0.011145)) if features[1] <= 1.500000 else -0.056432)
    raw_score += (((0.006606 if features[4] <= 0.474937 else -0.077055) if features[2] <= 24.500000 else (0.106817 if features[4] <= 0.469669 else 0.018406)) if features[1] <= 1.500000 else -0.056080)
    raw_score += (((-0.068753 if features[1] <= 0.000000 else -0.074396) if features[0] <= 9.500000 else (0.099080 if features[4] <= 0.236111 else 0.009971)) if features[1] <= 1.500000 else -0.055748)
    raw_score += (((0.011201 if features[0] <= 698.000000 else -0.074651) if features[0] <= 713.000000 else (0.098630 if features[0] <= 714.500000 else 0.103208)) if features[1] <= 1.500000 else (-0.055436 if features[0] <= 623.500000 else -0.055436))
    raw_score += (((0.045345 if features[4] <= 0.449495 else 0.013240) if features[1] <= 0.000000 else (0.015970 if features[4] <= 0.325758 else -0.038268)) if features[1] <= 1.500000 else (-0.055143 if features[0] <= 470.500000 else -0.055143))
    raw_score += (((-0.068525 if features[1] <= 0.000000 else -0.073270) if features[0] <= 9.500000 else (0.092815 if features[4] <= 0.236111 else 0.008407)) if features[1] <= 1.500000 else -0.054867)
    raw_score += (((0.004948 if features[0] <= 18.500000 else -0.072765) if features[2] <= 4.500000 else (0.071725 if features[0] <= 65.000000 else 0.009240)) if features[1] <= 1.500000 else (-0.054607 if features[0] <= 671.500000 else (-0.054607 if features[0] <= 679.500000 else -0.054607)))
    # Sigmoid function
    return 1.0 / (1.0 + math.exp(-raw_score))
# =====================================================================
# 1. LightGBM 轉譯出來的純 Python 預測器 (直接內嵌，零依賴)
# =====================================================================



# =====================================================================
# 2. LightGBM 驅動的自適應風控引擎
# =====================================================================
class LGBMConfidenceGateEngine:
    def __init__(
        self, 
        window_size: int = 4, 
        lookback_horizon: int = 240, 
        tolerance: int = 2,
        gamma: float = 1.8,
        ml_threshold: float = 0.55  # ML 置信度門檻 (>=55% 採信長線訊號)
    ):
        self.window_size = window_size
        self.lookback_horizon = lookback_horizon
        self.gamma = gamma
        self.tolerance = tolerance
        self.ml_threshold = ml_threshold

        self.observed_events: Set[Tuple[int, str]] = set()
        self.past_predictions: Dict[Tuple[int, str], int] = {}

        self.telemetry = {
            "accepted_long_signals": 0,
            "rejected_long_signals": 0,
            "immediate_fallbacks": 0,
        }

    def update_actual_events(self, current_step: int, observed_items: List[str]):
        for item in observed_items:
            self.observed_events.add((current_step, item))

    def get_gated_forecast(
        self, 
        current_step: int, 
        item: str, 
        raw_predictions: Dict[Tuple[int, str], int]
    ) -> int:
        # 1. 取得模型原始發單總量
        raw_signal = sum(raw_predictions.get((current_step + d, item), 0) for d in range(1, self.window_size + 1))
        
        if raw_signal == 0:
            return 0

        # 2. 提取特徵並計算 LightGBM 預測機率 P(hit)
        feature_vector = self._extract_features(current_step, item, raw_signal)
        win_prob = predict_lgb_probability(feature_vector)  # 產出 0.0 ~ 1.0 之間的機率

        # 3. Dynamic Sizing 期望值最大化公式
        # Effective Qty = Round( Raw_Signal * P(hit)^gamma )
        confidence_multiplier = win_prob ** self.gamma
        effective_qty = int(round(raw_signal * confidence_multiplier))

        return effective_qty