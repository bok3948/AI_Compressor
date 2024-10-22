# AI_Compressor

Neural Network Compressor with Pruning, Knowledge Distillation, and Quantization.

## Overview
AI_Compressor is a tool for compressing neural network models using various techniques, including pruning, knowledge distillation (KD), and quantization. It provides two modes of operation to accommodate different use cases, whether training data is available or not.

### Modes of Operation
- **Mode 0 (Post-training Setting)**: For scenarios where training data is not available and retraining is not feasible. Requires only a calibration dataset to achieve compression.
- **Mode 1 (Training Available)**: Designed for scenarios where training data is available, allowing further model optimization through knowledge distillation and quantization-aware training (QAT).

![Compression Modes](./Images/mode1.png "Mode Overview")

## How to Run
To run the compressor, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/AI_Compressor.git
   cd AI_Compressor
   ```

2. Install the necessary dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the compressor in your desired mode:
   ```sh
   python compressor.py --mode [0 or 1] --model [MODEL_NAME] --data [DATA_PATH]
   ```
   - **mode**: Choose either `0` or `1` depending on the availability of training data.
   - **model**: Specify the model to be compressed (e.g., MobileNetV2, ResNet18, DeiT).
   - **data**: Provide the path to the dataset (required for Mode 1).

## Results
The compressor has been tested with various models, including MobileNetV2, ResNet18, and DeiT. The results demonstrate a trade-off between latency, size, and accuracy, effectively balancing these metrics using different compression techniques.

![Model Compression Results](./Images/result.png "Results Overview")

## Reproduce
- To run MobileNetV2 compression in post-training mode:
  ```sh
  python compressor.py --mode 0 --model MobileNetV2 --data ./data/calibration
  ```
- To run DeiT compression with training:
  ```sh
  python compressor.py --mode 1 --model DeiT --data ./data/train
  ```




