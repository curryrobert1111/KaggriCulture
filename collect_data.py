import csv
import math
import numpy as np
from typing import Dict, List, Set, Tuple


class AdvancedFeatureExtractor:
    def __init__(self, ema_span: int = 20):
        self.ema_alpha = 2.0 / (ema_span + 1)
        self.ema_hits: Dict[str, float] = {"WHEAT": 0.5, "HERD_A": 0.5, "HERD_B": 0.5}
        self.last_event_step: Dict[str, int] = {"WHEAT": -999, "HERD_A": -999, "HERD_B": -999}

    def update_history(self, current_step: int, observed_items: List[str]):
        """每回合更新 EMA 與跨商品最後出現時間"""
        for item in ["WHEAT", "HERD_A", "HERD_B"]:
            is_hit = 1.0 if item in observed_items else 0.0
            self.ema_hits[item] = (self.ema_alpha * is_hit) + ((1.0 - self.ema_alpha) * self.ema_hits[item])
            if is_hit:
                self.last_event_step[item] = current_step

    def get_feature_dict(self, current_step: int, item: str, raw_signal: int) -> dict:
        period = 720.0
        sin_step = math.sin(2 * math.pi * (current_step % period) / period)
        cos_step = math.cos(2 * math.pi * (current_step % period) / period)

        wheat_delta = current_step - self.last_event_step["WHEAT"]
        is_wheat_recent = 1 if wheat_delta <= 10 else 0

        ema_hit_rate = self.ema_hits.get(item, 0.5)
        item_code = {"WHEAT": 0, "HERD_A": 1, "HERD_B": 2}.get(item, 0)

        return {
            "current_step": current_step,
            "sin_step": sin_step,
            "cos_step": cos_step,
            "item_code": item_code,
            "raw_signal": raw_signal,
            "ema_hit_rate": ema_hit_rate,
            "wheat_delta": wheat_delta,
            "is_wheat_recent": is_wheat_recent,
            "last_event_step": self.last_event_step[item]
        }


class DataCollectorGateEngine:
    def __init__(self, window_size: int = 4, tolerance: int = 2):
        self.window_size = window_size
        self.tolerance = tolerance
        self.observed_events: Set[Tuple[int, str]] = set()
        self.extractor = AdvancedFeatureExtractor()
        self.pending_samples: List[dict] = []

    def update_actual_events(self, current_step: int, observed_items: List[str]):
        for item in observed_items:
            self.observed_events.add((current_step, item))
        self.extractor.update_history(current_step, observed_items)

    def record_signal_features(
        self, 
        current_step: int, 
        item: str, 
        raw_predictions: Dict[Tuple[int, str], int]
    ) -> int:
        extended_signal = sum(raw_predictions.get((current_step + d, item), 0) for d in range(1, self.window_size + 1))

        if extended_signal < 2:
            return extended_signal

        target_step = current_step + 3
        features = self.extractor.get_feature_dict(current_step, item, extended_signal)

        self.pending_samples.append({
            "target_step": target_step,
            "item": item,
            "features": features
        })

        return extended_signal

    def finalize_and_get_dataset(self) -> List[dict]:
        dataset = []
        match_deltas = range(-self.tolerance, self.tolerance + 1)

        for sample in self.pending_samples:
            t_target = sample["target_step"]
            item = sample["item"]
            
            is_hit = 1 if any((t_target + delta, item) in self.observed_events for delta in match_deltas) else 0
            
            row = sample["features"].copy()
            row["label"] = is_hit
            dataset.append(row)
            
        return dataset


def run_data_collection(num_games: int = 200, output_file: str = "dataset.csv"):
    all_rows = []
    print(f"=== 開始採集 {num_games} 局「高多樣性隨機對戰」特徵 ===")

    for seed in range(num_games):
        collector = DataCollectorGateEngine()
        rng = np.random.RandomState(seed)
        
        # 隨機產生每局獨特的事件週期與雜訊比率
        wheat_interval = rng.randint(8, 14)
        herd_interval = rng.randint(12, 18)
        noise_rate = rng.uniform(0.15, 0.45)

        for step in range(720):
            actual_sales = []
            if step % wheat_interval == 0:
                actual_sales.append("WHEAT")
            if step % herd_interval == 0:
                actual_sales.append("HERD_A")
            
            collector.update_actual_events(step, actual_sales)

            preds = {}
            if step % (wheat_interval - 2) == 0:
                preds[(step + 3, "WHEAT")] = 2
            if step % (herd_interval - 2) == 0:
                preds[(step + 3, "HERD_A")] = 2
                
            if rng.rand() < noise_rate:
                fake_item = rng.choice(["WHEAT", "HERD_A", "HERD_B"])
                preds[(step + 3, fake_item)] = 2

            for item in ["WHEAT", "HERD_A", "HERD_B"]:
                collector.record_signal_features(step, item, preds)

        all_rows.extend(collector.finalize_and_get_dataset())

    if not all_rows:
        print("未採集到數據！")
        return

    fieldnames = list(all_rows[0].keys())
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"採集完成！成功生成 {len(all_rows)} 筆『高多樣性』樣本並寫入 {output_file}")


if __name__ == "__main__":
    run_data_collection(num_games=200)