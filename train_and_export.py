import pandas as pd
import lightgbm as lgb


# 1. 轉譯器：將 LightGBM 的 Tree 結構轉成純 Python 函數代碼
def tree_to_python_code(node):
    if 'leaf_value' in node:
        return f"{node['leaf_value']:.6f}"
    
    feat_idx = node['split_feature']
    thresh = node['threshold']
    left_code = tree_to_python_code(node['left_child'])
    right_code = tree_to_python_code(node['right_child'])
    
    return f"({left_code} if features[{feat_idx}] <= {thresh:.6f} else {right_code})"


def generate_pure_python_predictor(booster, feature_names):
    model_dict = booster.dump_model()
    tree_exprs = [tree_to_python_code(t['tree_structure']) for t in model_dict['tree_info']]
    
    code = f"# 特徵順序 ({len(feature_names)} 維): {feature_names}\n"
    code += "import math\n\n"
    code += "def predict_lgb_probability(features: list) -> float:\n"
    code += "    raw_score = 0.0\n"
    for expr in tree_exprs:
        code += f"    raw_score += {expr}\n"
    code += "    # Sigmoid function\n"
    code += "    return 1.0 / (1.0 + math.exp(-raw_score))\n"
    return code


# 2. 訓練與匯出流程
def train_and_export():
    df = pd.read_csv("dataset.csv")
    
    # 對齊 9 維進階特徵欄位順序
    feature_cols = [
        "current_step", "sin_step", "cos_step", "item_code", 
        "raw_signal", "ema_hit_rate", "wheat_delta", 
        "is_wheat_recent", "last_event_step"
    ]
    
    X = df[feature_cols]
    y = df["label"]

    print(f"訓練樣本數量: {len(df)} | 正樣本比例: {y.mean():.2%}")

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 7,        # 輕量化樹節點
        "max_depth": 3,
        "verbose": -1
    }

    gbm = lgb.train(params, train_data, num_boost_round=25)

    # 3. 導出為純 Python 代碼並寫入 generated_model.py
    py_code = generate_pure_python_predictor(gbm, feature_cols)
    with open("generated_model.py", "w") as f:
        f.write(py_code)

    print("LightGBM 模型成功訓練並導出至 'generated_model.py'！")


if __name__ == "__main__":
    train_and_export()