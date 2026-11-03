# TinyML — Speech Recognition on Arduino

Speech-command recognition on an Arduino Nano 33 BLE, exploring the tradeoffs between accuracy, model size, and on-device processing time.

## Technical scope

- Audio preprocessing and visualization: waveforms, spectra, spectrograms, mel features, and MFCCs.
- Model training and analysis with PyTorch.
- Model conversion and deployment using TensorFlow Lite Micro.
- Post-training quantization and quantization-aware training across bit widths.
- Structured and unstructured pruning, with fine-tuning and accuracy comparisons.

## Recorded results

Approximately 90% offline accuracy and 110 ms total on-device processing time. The saved report contains differing breakdowns of preprocessing and inference time that both total 110 ms. Offline accuracy should not be interpreted as measured field accuracy on the microcontroller.

## Deployment

See [Arduino Nano 33 BLE deployment instructions](arduino_nano_33_ble_tutorial.md). The notebooks document preprocessing, training, conversion, quantization, and pruning experiments.

## Attribution

Built on the ML-HW-SYS starter implementation and experimental framework. Existing source credits and license notices are retained.
