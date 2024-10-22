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
   git clone https://github.com/bok3948/AI_Compressor.git
   cd AI_Compressor
   ```

2. Install the necessary dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the compressor :
   ```sh
   python compressor.py --mode [0 or 1] --model [MODEL_NAME] --data [DATA_PATH] 
   ```
   #available model list: resnet, deit(from timm), mobilenetv2 (other model may be not compatible with this code) Most of CNN model is okay but transoformer need to be change forward function 

## Results
The compressor has been tested with various models, including MobileNetV2, ResNet18, and DeiT. The results demonstrate a trade-off between latency, size, and accuracy.

![Model Compression Results](./Images/resullt.png "Results Overview")

## Reproduce
- DownLoad Trained Model here [Google Drive Link](https://drive.google.com/file/d/1OmCzW_q9zCORb38RHm528-AKhN86Zeli/view?usp=drive_link)
- To run MobileNetV2 compression mode:
  ```sh
  python main.py --device cuda --mode 1 --data_set CIFAR --data_path ./ --model mobilenetv2_x1_0 --pretrained ./cifar100_mobilenetv2_x1_0 --pruning_ratio 0.6 --global_pruning True --weight_decay 0.0005 --lr 1e-6 --qat_lr 1e-6 --qat_epochs 10 --epochs 100 --do_KD
  ```


  






