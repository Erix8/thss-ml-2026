import numpy as np
import pandas as pd

CONFIG = {
    "num_samples": 1100,
    "seed": 42,
    "output_csv": "train.csv",
    "output_csv_test": "test.csv",

    "x_specs": {
        "x1": {"dist": "normal",  "mean": 0.0, "var": 1.0},
        "x2": {"dist": "normal",  "mean": 2.0, "var": 4.0},
        "x3": {"dist": "uniform", "low": -3.0, "high": 3.0},
        "x4": {"dist": "normal",  "mean": -1.0, "var": 0.25},
        "x5": {"dist": "uniform", "low": 0.0, "high": 5.0},
        
        "x6": {"dist": "normal", "mean": 3.2, "var": 6.8},
        "x7": {"dist": "uniform", "low": -7.5, "high": 4.1},
        "x8": {"dist": "normal", "mean": -2.9, "var": 3.4},
        "x9": {"dist": "uniform", "low": -9.2, "high": 1.5},
        "x10": {"dist": "normal", "mean": 5.7, "var": 8.2},

        "x11": {"dist": "normal", "mean": 300.0, "var": 1500.0},

        "x12": {"dist": "uniform", "low": -6.3, "high": 7.8},
        "x13": {"dist": "normal", "mean": -4.5, "var": 2.1},
        "x14": {"dist": "uniform", "low": -2.1, "high": 8.5},
        "x15": {"dist": "normal", "mean": 1.8, "var": 4.9},
        "x16": {"dist": "uniform", "low": -8.7, "high": 2.3},
        "x17": {"dist": "normal", "mean": -6.2, "var": 5.3},
        "x18": {"dist": "uniform", "low": -5.4, "high": 9.1},

        "x19": {"dist": "normal", "mean": 500.0, "var": 2500.0},

        "x20": {"dist": "normal", "mean": 2.5, "var": 7.6},
        "x21": {"dist": "uniform", "low": -1.8, "high": 6.7},
        "x22": {"dist": "normal", "mean": -3.7, "var": 1.9},
        "x23": {"dist": "uniform", "low": -9.5, "high": 0.8},
        "x24": {"dist": "normal", "mean": 7.1, "var": 3.8},

        "x25": {"dist": "normal", "mean": 1200.0, "var": 8000.0},

        "x26": {"dist": "uniform", "low": -4.2, "high": 5.9},
        "x27": {"dist": "normal", "mean": -1.3, "var": 6.2},
        "x28": {"dist": "uniform", "low": -7.1, "high": 3.6},
        "x29": {"dist": "normal", "mean": 4.9, "var": 2.7},

        "x30": {"dist": "uniform", "low": 200.0, "high": 600.0},

        "x31": {"dist": "normal", "mean": -5.8, "var": 4.5},
        "x32": {"dist": "uniform", "low": -3.5, "high": 8.2},
        "x33": {"dist": "normal", "mean": 6.3, "var": 7.1},

        "x34": {"dist": "uniform", "low": 800.0, "high": 1500.0},

        "x35": {"dist": "normal", "mean": -2.4, "var": 5.6},
        
    },

    # Relationship between y and x (customizable coefficients)
    # y = bias + Σ(linear_coef_i * x_i) + Σ(nonlinear_term) + noise
    #
    # Supported nonlinear terms:
    # - square: coef * x^2
    # - cube: coef * x^3
    # - sin: coef * sin(x)
    # - cos: coef * cos(x)
    # - interaction: coef * x_a * x_b
    "y_specs": {
        "y1": {
            "bias": -0.5,
            "linear": {"x1": -2.0},
            "nonlinear": [
                {"type": "cos", "x": "x2", "coef": 0.5},
                {"type": "square", "x": "x3", "coef": 0.2},
            ],
            "noise_std": 0.05,
        },
        "y2": {
            "bias": 0.5,
            "linear": {
                "x1": 0.12, "x2": -0.08, "x3": 0.15, "x4": -0.07, "x5": 0.09,
                "x6": -0.11, "x7": 0.06, "x8": -0.13, "x9": 0.04, "x10": -0.05,
                "x11": 0.0013, "x12": -0.09, "x13": 0.07, "x14": -0.06, "x15": 0.10,
                "x16": -0.04, "x17": 0.08, "x18": -0.12, "x19": 0.005, "x20": -0.03,
                "x21": 0.06, "x22": -0.10, "x23": 0.09, "x24": -0.07, "x25": 0.002,
                "x26": -0.05, "x27": 0.11, "x28": -0.04, "x29": 0.08, "x30": -0.006,
                "x31": 0.03, "x32": -0.09, "x33": 0.05, "x34": -0.002, "x35": 0.07,
            },
            "noise_std": 0.20,
        },
        "y3": {
            "task": "multiclass",
            "num_classes": 4,
            "class_params": [
                # Class 0: expected score ~ 0.850
                {"bias": 0.20, "linear": {
                    "x1": 0.1556, "x3": 0.2292, "x5": 0.1110, "x6": -0.1597, "x8": -0.2221,
                    "x9": 0.0592, "x10": -0.0702, "x13": 0.1020, "x14": -0.0805, "x17": 0.1266,
                    "x18": -0.1665, "x19": 0.0050, "x22": -0.1536, "x23": 0.1249, "x24": -0.0970,
                    "x25": 0.0020, "x26": -0.0650, "x27": 0.1605, "x31": 0.0403, "x32": -0.1267,
                    "x33": 0.0702, "x34": -0.0010
                }},
                # Class 1: expected score ~ 0.850
                {"bias": -0.10, "linear": {
                    "x2": 0.1220, "x3": -0.0801, "x4": 0.1741, "x5": -0.0493, "x6": 0.0925,
                    "x7": -0.1427, "x8": 0.0750, "x9": -0.1055, "x10": 0.0421, "x11": -0.0012,
                    "x13": -0.0583, "x14": 0.1207, "x15": -0.0970, "x16": 0.0270, "x17": -0.0790,
                    "x19": -0.0030, "x20": 0.1067, "x23": -0.1381, "x24": 0.0970, "x25": -0.0020,
                    "x26": 0.1170, "x27": -0.0421, "x29": -0.1547, "x30": 0.0040, "x34": 0.0090,
                    "x35": -0.0842
                }},
                # Class 2: expected score ~ 0.850
                {"bias": 0.70, "linear": {
                    "x1": 0.0648, "x4": -0.0402, "x5": 0.1726, "x6": -0.0871, "x7": 0.1284,
                    "x8": -0.0300, "x9": 0.1319, "x10": -0.0983, "x12": -0.2217, "x13": 0.0291,
                    "x16": -0.1349, "x17": 0.0474, "x19": 0.0080, "x20": -0.1467, "x21": 0.0134,
                    "x22": -0.0921, "x23": 0.1665, "x24": -0.0554, "x26": -0.0260, "x27": 0.1312,
                    "x29": 0.0421, "x30": -0.0080, "x31": 0.0672, "x32": -0.0563, "x33": 0.1404,
                    "x35": 0.0281
                }},
                # Class 3: expected score ~ 0.850
                {"bias": -0.40, "linear": {
                    "x3": -0.1833, "x4": 0.0670, "x5": -0.0987, "x7": -0.0999, "x8": 0.1500,
                    "x10": 0.1544, "x11": -0.0050, "x12": 0.1364, "x13": -0.1601, "x14": 0.0403,
                    "x15": -0.0554, "x16": 0.0944, "x17": -0.1422, "x18": 0.0278, "x20": 0.0533,
                    "x21": -0.1946, "x23": -0.1105, "x24": 0.1249, "x25": -0.0030, "x26": 0.0780,
                    "x27": -0.0140, "x28": 0.0694, "x32": 0.0985, "x33": -0.0842, "x34": 0.0030,
                    "x35": -0.1403
                }}
            ],
            "noise_std": 0.1
        },
    }
}


# =========================================================
# 2) Generate X
# =========================================================

def sample_one_feature(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    dist = spec["dist"]

    if dist == "normal":
        mean = float(spec.get("mean", 0.0))
        if "std" in spec:
            std = float(spec["std"])
        else:
            var = float(spec.get("var", 1.0))
            if var < 0:
                raise ValueError("var of normal distribution must be non-negative")
            std = np.sqrt(var)
        return rng.normal(loc=mean, scale=std, size=n)

    elif dist == "uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        if high <= low:
            raise ValueError("uniform distribution requires high > low")
        return rng.uniform(low=low, high=high, size=n)

    else:
        raise ValueError(f"unsupported distribution type: {dist}")


def generate_x(config: dict) -> dict:
    n = int(config["num_samples"])
    rng = np.random.default_rng(int(config["seed"]))

    x_data = {}
    for x_name, x_spec in config["x_specs"].items():
        x_data[x_name] = sample_one_feature(x_spec, n, rng)

    return x_data, rng


# =========================================================
# 3) Generate Y
# =========================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z, temperature=2):
    # z: (n_samples, n_classes)
    z = z / temperature
    z = z - np.max(z, axis=1, keepdims=True)  # for numerical stability
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def build_continuous_score(y_spec: dict, x_data: dict, n: int):
    score = np.zeros(n, dtype=float)
    score += float(y_spec.get("bias", 0.0))

    for x_name, coef in y_spec.get("linear", {}).items():
        score += float(coef) * x_data[x_name]

    for term in y_spec.get("nonlinear", []):
        score += apply_nonlinear_term(term, x_data)

    return score

def apply_nonlinear_term(term: dict, x_data: dict) -> np.ndarray:
    t = term["type"]
    coef = float(term.get("coef", 1.0))

    if t == "square":
        x = x_data[term["x"]]
        return coef * (x ** 2)

    elif t == "cube":
        x = x_data[term["x"]]
        return coef * (x ** 3)

    elif t == "sin":
        x = x_data[term["x"]]
        return coef * np.sin(x)

    elif t == "cos":
        x = x_data[term["x"]]
        return coef * np.cos(x)

    elif t == "interaction":
        xa = x_data[term["x1"]]
        xb = x_data[term["x2"]]
        return coef * (xa * xb)

    else:
        raise ValueError(f"unsupported nonlinear term type: {t}")


def generate_y(config: dict, x_data: dict, rng: np.random.Generator) -> dict:
    n = int(config["num_samples"])
    y_data = {}

    for y_name, y_spec in config["y_specs"].items():
        task = y_spec.get("task", "regression")  # default: regression

        if task == "regression":
            y = build_continuous_score(y_spec, x_data, n)
            noise_std = float(y_spec.get("noise_std", 0.0))
            if noise_std > 0:
                y += rng.normal(0.0, noise_std, size=n)
            y_data[y_name] = y.astype(float)

        elif task == "binary":
            score = build_continuous_score(y_spec, x_data, n)
            noise_std = float(y_spec.get("noise_std", 0.0))
            if noise_std > 0:
                score += rng.normal(0.0, noise_std, size=n)

            p = sigmoid(score)

            # sample by probability (like real classification data)
            y = rng.binomial(1, p, size=n)

            y_data[y_name] = y.astype(int)

        elif task == "multiclass":
            k = int(y_spec["num_classes"])
            class_params = y_spec["class_params"]
            if len(class_params) != k:
                raise ValueError(f"{y_name}: class_params length must equal num_classes")

            scores = np.zeros((n, k), dtype=float)
            for c in range(k):
                cp = class_params[c]
                s = np.zeros(n, dtype=float) + float(cp.get("bias", 0.0))
                for x_name, coef in cp.get("linear", {}).items():
                    s += float(coef) * x_data[x_name]
                # To add per-class nonlinear terms, use cp.get("nonlinear", [])
                scores[:, c] = s

            noise_std = float(y_spec.get("noise_std", 0.0))
            if noise_std > 0:
                scores += rng.normal(0.0, noise_std, size=scores.shape)

            scores = scores - np.array([0.864, 10.548, -0.463, 0.729])
            probs = softmax(scores)

            # sample by class probability
            y = np.array([rng.choice(k, p=probs[i]) for i in range(n)], dtype=int)
            y_data[y_name] = y

        else:
            raise ValueError(f"{y_name}: unsupported task type {task}")

    return y_data


# =========================================================
# 4) Save CSV
# =========================================================

def main():
    x_data, rng = generate_x(CONFIG)
    y_data = generate_y(CONFIG, x_data, rng)

    # sort columns as x1,x2,...,y1,y2,...
    x_cols = sorted(x_data.keys(), key=lambda s: int(s[1:]))
    y_cols = sorted(y_data.keys(), key=lambda s: int(s[1:]))

    df = pd.DataFrame({**{k: x_data[k] for k in x_cols},
                    **{k: y_data[k] for k in y_cols}})

    output_csv = CONFIG["output_csv"]
    output_csv_test = CONFIG["output_csv_test"]

    df_train = df.head(1000)  # first file: 1000 rows
    df_test = df.tail(100)    # second file: 100 rows

    df_train.to_csv(output_csv, index=False, encoding="utf-8")
    df_test.to_csv(output_csv_test, index=False, encoding="utf-8")

    print(f"Training data (1000 rows) saved to: {output_csv}")
    print(f"Test data (100 rows) saved to: {output_csv_test}")
    print("Total rows:", len(df))
    print("Training set rows:", len(df_train))
    print("Test set rows:", len(df_test))
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows of training set:")
    print(df_train.head())


if __name__ == "__main__":
    main()