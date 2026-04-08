# TinyML Smart Load Scheduler (ESP32, Fully Edge-Based)

This repository contains a complete TinyML workflow for an ESP32-based smart load scheduler:

1. Simulate/load training data on-device context (`time`, `motion`, `light`, `previous_state`, `duration`, `current_feedback`).
2. Train an ultra-small sequence classifier in TensorFlow/Keras.
3. Convert to INT8 TensorFlow Lite and export to `model.h`.
4. Run inference fully on ESP32 with TensorFlow Lite for Microcontrollers (no cloud).

## Project Structure

- `python/train_tinyml_model.py`: dataset simulation, preprocessing, sliding windows, train/test split, model training, accuracy report.
- `python/convert_to_tflite.py`: INT8 quantization, `.tflite` export, C header export.
- `firmware/esp32_smart_load_scheduler.ino`: full ESP32 firmware (RTC + PIR + LDR + ACS712 + relay + TFLM inference).
- `firmware/model.h`: example header (replace with generated header after conversion).

## Quick Start

1. Install Python dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn
```

Use Python 3.11 or 3.12 for TensorFlow compatibility.

2. Train model:

```bash
python python/train_tinyml_model.py
```

3. Convert and export:

```bash
python python/convert_to_tflite.py
```

4. Copy generated header to firmware location:

```bash
cp artifacts/model.h firmware/model.h
```

5. Build/upload `firmware/esp32_smart_load_scheduler.ino` in Arduino IDE (ESP32 board).

## Notes

- Inference is fully local on ESP32.
- Conversion script enforces a strict model-size gate (`<= 2 KB`) and exits with failure if exceeded.
- Update scaler constants in firmware from `artifacts/scaler_params.npz` for best accuracy.
