/*
  TinyML Smart Load Scheduler (ESP32)
  -----------------------------------
  Fully edge-based inference using TensorFlow Lite for Microcontrollers.

  Inputs:
    - RTC time_of_day (normalized)
    - PIR motion
    - LDR light level
    - previous load state
    - duration (state persistence)
    - ACS712 current feedback

  Output:
    - Decision class: ON / OFF / DELAY
    - Relay control signal

  Workflow:
    1) Read sensors
    2) Build feature vector
    3) Update 10-step sliding window
    4) Run TinyML inference
    5) Apply decision logic
    6) Closed-loop feedback check using current sensor
*/

#include <Arduino.h>
#include <Wire.h>
#include <RTClib.h>

#include <TensorFlowLite.h>
#include "model.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// -----------------------------
// Hardware Pins (adjust as needed)
// -----------------------------
constexpr int PIR_PIN = 27;
constexpr int LDR_PIN = 34;
constexpr int ACS_PIN = 35;
constexpr int RELAY_PIN = 26;

// Relay wiring: set true if HIGH means ON for your relay module.
constexpr bool RELAY_ACTIVE_HIGH = true;

// -----------------------------
// Feature/Model Constants
// -----------------------------
constexpr int kSequenceLength = 10;
constexpr int kFeatureCount = 6;
constexpr int kNumClasses = 3;

// Keep tensor arena minimal for tiny model.
constexpr size_t kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

enum DecisionClass : int {
  DECISION_ON = 0,
  DECISION_OFF = 1,
  DECISION_DELAY = 2,
};

// Copy values from artifacts/scaler_params.npz after training for best performance.
// These defaults are safe placeholders.
float kFeatureMean[kFeatureCount] = {0.5f, 0.4f, 0.5f, 0.5f, 0.2f, 0.2f};
float kFeatureStd[kFeatureCount] = {0.29f, 0.49f, 0.28f, 0.50f, 0.21f, 0.24f};

// -----------------------------
// Sensor/Control State
// -----------------------------
RTC_DS3231 rtc;

bool relay_state = false;
unsigned long state_start_ms = 0;

float feature_history[kSequenceLength][kFeatureCount];
int history_count = 0;

unsigned long last_sample_ms = 0;
constexpr unsigned long SAMPLE_INTERVAL_MS = 1000;  // 1 second

// -----------------------------
// TensorFlow Lite Micro State
// -----------------------------
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroMutableOpResolver<2> resolver;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// -----------------------------
// Utilities
// -----------------------------
float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

int argmax3(const float probs[kNumClasses]) {
  int best = 0;
  for (int i = 1; i < kNumClasses; ++i) {
    if (probs[i] > probs[best]) best = i;
  }
  return best;
}

const char* decisionToString(int d) {
  if (d == DECISION_ON) return "ON";
  if (d == DECISION_OFF) return "OFF";
  return "DELAY";
}

void setRelay(bool on) {
  relay_state = on;
  int level = RELAY_ACTIVE_HIGH ? (on ? HIGH : LOW) : (on ? LOW : HIGH);
  digitalWrite(RELAY_PIN, level);
}

// For ACS712-05B with 3.3V ADC path. Adjust sensitivity/reference for your module.
float readCurrentAmps() {
  int raw = analogRead(ACS_PIN);
  float voltage = (static_cast<float>(raw) / 4095.0f) * 3.3f;
  float current = (voltage - 1.65f) / 0.185f;
  return fabsf(current);
}

float readLightNorm() {
  int raw = analogRead(LDR_PIN);
  return clampf(static_cast<float>(raw) / 4095.0f, 0.0f, 1.0f);
}

float readTimeNorm() {
  DateTime now = rtc.now();
  int sec_of_day = now.hour() * 3600 + now.minute() * 60 + now.second();
  return static_cast<float>(sec_of_day) / 86400.0f;
}

// -----------------------------
// Feature Engineering
// -----------------------------
void pushFeatureVector(const float feat[kFeatureCount]) {
  if (history_count < kSequenceLength) {
    for (int j = 0; j < kFeatureCount; ++j) {
      feature_history[history_count][j] = feat[j];
    }
    history_count++;
    return;
  }

  for (int i = 0; i < kSequenceLength - 1; ++i) {
    for (int j = 0; j < kFeatureCount; ++j) {
      feature_history[i][j] = feature_history[i + 1][j];
    }
  }

  for (int j = 0; j < kFeatureCount; ++j) {
    feature_history[kSequenceLength - 1][j] = feat[j];
  }
}

bool fillInputTensorFromHistory() {
  if (!input_tensor) return false;
  if (history_count < kSequenceLength) return false;
  if (input_tensor->type != kTfLiteInt8) return false;

  const float in_scale = input_tensor->params.scale;
  const int in_zero_point = input_tensor->params.zero_point;

  int idx = 0;
  for (int t = 0; t < kSequenceLength; ++t) {
    for (int f = 0; f < kFeatureCount; ++f) {
      float normalized = (feature_history[t][f] - kFeatureMean[f]) / kFeatureStd[f];
      int quant = static_cast<int>(roundf(normalized / in_scale) + in_zero_point);
      quant = static_cast<int>(clampf(static_cast<float>(quant), -128.0f, 127.0f));
      input_tensor->data.int8[idx++] = static_cast<int8_t>(quant);
    }
  }

  return true;
}

bool runInference(float probs[kNumClasses], int& predicted_class) {
  if (!fillInputTensorFromHistory()) return false;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return false;
  }

  if (output_tensor->type != kTfLiteInt8) {
    TF_LITE_REPORT_ERROR(error_reporter, "Expected int8 output tensor");
    return false;
  }

  const float out_scale = output_tensor->params.scale;
  const int out_zero_point = output_tensor->params.zero_point;

  for (int i = 0; i < kNumClasses; ++i) {
    int8_t q = output_tensor->data.int8[i];
    probs[i] = (static_cast<int>(q) - out_zero_point) * out_scale;
  }

  predicted_class = argmax3(probs);
  return true;
}

// -----------------------------
// Decision + Closed-loop validation
// -----------------------------
void applyDecision(int decision) {
  bool previous = relay_state;

  if (decision == DECISION_ON) {
    setRelay(true);
  } else if (decision == DECISION_OFF) {
    setRelay(false);
  } else {
    // DELAY: keep current relay state
  }

  if (relay_state != previous) {
    state_start_ms = millis();
  }
}

void validateLoadResult() {
  float current_a = readCurrentAmps();
  const float on_threshold = 0.15f;
  const float off_threshold = 0.08f;

  bool valid = relay_state ? (current_a > on_threshold) : (current_a < off_threshold);

  Serial.print("Closed-loop current (A): ");
  Serial.print(current_a, 3);
  Serial.print(" -> ");
  Serial.println(valid ? "Load response OK" : "Load response MISMATCH");
}

void printSensorDebug(float time_norm, int motion, float light, float duration_norm, float current_norm) {
  Serial.print("Sensors | time=");
  Serial.print(time_norm, 3);
  Serial.print(", motion=");
  Serial.print(motion);
  Serial.print(", light=");
  Serial.print(light, 3);
  Serial.print(", prev_state=");
  Serial.print(relay_state ? 1 : 0);
  Serial.print(", duration=");
  Serial.print(duration_norm, 3);
  Serial.print(", current_norm=");
  Serial.println(current_norm, 3);
}

void printPredictionDebug(const float probs[kNumClasses], int decision) {
  // For this ultra-small model, output values are class scores (logits), not normalized probabilities.
  Serial.print("Prediction(logits) | ON=");
  Serial.print(probs[DECISION_ON], 4);
  Serial.print(", OFF=");
  Serial.print(probs[DECISION_OFF], 4);
  Serial.print(", DELAY=");
  Serial.print(probs[DECISION_DELAY], 4);
  Serial.print(" => ");
  Serial.println(decisionToString(decision));
}

// -----------------------------
// Arduino Setup/Loop
// -----------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(PIR_PIN, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  analogReadResolution(12);
  setRelay(false);
  state_start_ms = millis();

  Wire.begin();
  if (!rtc.begin()) {
    Serial.println("RTC not found. Check wiring.");
    while (true) delay(1000);
  }
  if (rtc.lostPower()) {
    // Initialize RTC with firmware compile time.
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  model = tflite::GetModel(g_smart_load_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch.");
    while (true) delay(1000);
  }

  // Ops for tiny model: MEAN (GlobalAveragePooling1D) + FULLY_CONNECTED.
  if (resolver.AddMean() != kTfLiteOk) {
    Serial.println("Resolver AddMean failed.");
    while (true) delay(1000);
  }
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("Resolver AddFullyConnected failed.");
    while (true) delay(1000);
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed.");
    while (true) delay(1000);
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("TinyML Smart Load Scheduler started.");
}

void loop() {
  unsigned long now_ms = millis();
  if (now_ms - last_sample_ms < SAMPLE_INTERVAL_MS) return;
  last_sample_ms = now_ms;

  float time_norm = readTimeNorm();
  int motion = digitalRead(PIR_PIN) == HIGH ? 1 : 0;
  float light_norm = readLightNorm();

  float duration_minutes = (now_ms - state_start_ms) / 60000.0f;
  float duration_norm = clampf(duration_minutes / 120.0f, 0.0f, 1.0f);

  float current_amps = readCurrentAmps();
  float current_norm = clampf(current_amps / 2.0f, 0.0f, 1.0f);

  float feat[kFeatureCount] = {
      time_norm,
      static_cast<float>(motion),
      light_norm,
      relay_state ? 1.0f : 0.0f,
      duration_norm,
      current_norm,
  };
  pushFeatureVector(feat);

  printSensorDebug(time_norm, motion, light_norm, duration_norm, current_norm);

  if (history_count < kSequenceLength) {
    Serial.print("Warm-up: collecting sequence window ");
    Serial.print(history_count);
    Serial.print("/");
    Serial.println(kSequenceLength);
    return;
  }

  float probs[kNumClasses] = {0};
  int decision = DECISION_DELAY;
  if (!runInference(probs, decision)) {
    Serial.println("Inference skipped due to tensor/input issue.");
    return;
  }

  printPredictionDebug(probs, decision);
  applyDecision(decision);

  Serial.print("Decision Applied -> Relay is now: ");
  Serial.println(relay_state ? "ON" : "OFF");

  delay(200);
  validateLoadResult();
  Serial.println("----");
}
