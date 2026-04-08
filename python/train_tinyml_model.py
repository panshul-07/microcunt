"""
Train a TinyML-friendly ultra-small model for edge smart load scheduling.

Part 1 + Part 2 from requested workflow:
- Dataset simulation/collection format
- Normalization and sliding windows (sequence length = 10)
- Train/test split
- Ultra-small sequence classifier training and accuracy report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "time_of_day",
    "motion",
    "light_level",
    "previous_state",
    "duration",
    "current_feedback",
]
LABEL_TO_ID = {"ON": 0, "OFF": 1, "DELAY": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
WINDOW_SIZE = 10


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def simulate_dataset(num_samples: int = 6000) -> pd.DataFrame:
    """
    Simulate behavior data.
    Output columns:
    - time_of_day (0..1)
    - motion (0/1)
    - light_level (0..1)
    - previous_state (0/1)
    - duration (0..1)
    - current_feedback (0..1)
    - label (ON/OFF/DELAY)
    """
    rows = []
    previous_state = 0
    state_duration_steps = 0

    for i in range(num_samples):
        minute_of_day = i % 1440
        hour = minute_of_day / 60.0
        time_of_day = minute_of_day / 1440.0

        daylight_base = 0.5 + 0.45 * np.sin((2.0 * np.pi * time_of_day) - np.pi / 2.0)
        light_level = clamp(daylight_base + np.random.normal(0, 0.08), 0.0, 1.0)

        motion_prob = 0.15
        if 6 <= hour <= 9 or 18 <= hour <= 23:
            motion_prob = 0.65
        elif 9 < hour < 18:
            motion_prob = 0.35
        motion = int(np.random.rand() < motion_prob)

        duration = clamp(state_duration_steps / 120.0, 0.0, 1.0)

        if previous_state == 1:
            current_feedback = clamp(np.random.normal(0.62, 0.08), 0.0, 1.0)
        else:
            current_feedback = clamp(np.random.normal(0.05, 0.03), 0.0, 1.0)

        if motion == 1 and light_level < 0.45:
            label = "ON"
        elif motion == 0 and previous_state == 1 and duration > 0.2:
            label = "OFF"
        elif motion == 1 and 0.45 <= light_level <= 0.70:
            label = "DELAY"
        else:
            label = "ON" if previous_state == 1 and motion == 1 else "OFF"

        rows.append(
            {
                "time_of_day": time_of_day,
                "motion": motion,
                "light_level": light_level,
                "previous_state": previous_state,
                "duration": duration,
                "current_feedback": current_feedback,
                "label": label,
            }
        )

        if label == "ON":
            next_state = 1
        elif label == "OFF":
            next_state = 0
        else:
            next_state = previous_state

        if next_state == previous_state:
            state_duration_steps += 1
        else:
            state_duration_steps = 0
        previous_state = next_state

    return pd.DataFrame(rows)


def build_sliding_windows(
    x: np.ndarray, y: np.ndarray, seq_len: int = WINDOW_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create windows of shape (seq_len, features) and assign label from window end.
    """
    x_windows = []
    y_windows = []
    for end in range(seq_len - 1, len(x)):
        start = end - seq_len + 1
        x_windows.append(x[start : end + 1])
        y_windows.append(y[end])
    return np.asarray(x_windows, dtype=np.float32), np.asarray(y_windows, dtype=np.int32)


def build_model(input_shape: tuple[int, int] = (10, 6)) -> tf.keras.Model:
    """
    Tiny architecture designed for strict model-size budgets:
    Input(10,6) -> GlobalAveragePooling1D -> Dense(3 logits)

    Why this is tiny:
    - Only one trainable layer (6*3 + 3 = 21 params)
    - Preserves sequence input requirement while minimizing FlatBuffer size.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(3),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    print("Generating dataset...")
    df = simulate_dataset(num_samples=6000)
    df.to_csv(ARTIFACTS_DIR / "simulated_dataset.csv", index=False)

    x_raw = df[FEATURE_COLUMNS].values.astype(np.float32)
    y_raw = df["label"].map(LABEL_TO_ID).values.astype(np.int32)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw).astype(np.float32)

    x_seq, y_seq = build_sliding_windows(x_scaled, y_raw, seq_len=WINDOW_SIZE)
    y_one_hot = tf.keras.utils.to_categorical(y_seq, num_classes=3)

    x_train, x_test, y_train, y_test = train_test_split(
        x_seq,
        y_one_hot,
        test_size=0.2,
        random_state=SEED,
        stratify=y_seq,
    )

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    model = build_model(input_shape=(WINDOW_SIZE, len(FEATURE_COLUMNS)))
    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
    )

    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    model.save(ARTIFACTS_DIR / "smart_load_cnn.keras")
    np.savez(
        ARTIFACTS_DIR / "scaler_params.npz",
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )
    np.save(ARTIFACTS_DIR / "representative_data.npy", x_train[:300].astype(np.float32))
    with open(ARTIFACTS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(ID_TO_LABEL, f, indent=2)

    print("Saved artifacts:")
    print("- artifacts/smart_load_cnn.keras")
    print("- artifacts/scaler_params.npz")
    print("- artifacts/representative_data.npy")
    print("- artifacts/label_map.json")
    print("- artifacts/simulated_dataset.csv")


if __name__ == "__main__":
    main()
