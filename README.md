# FashionMNIST CNN Classifier

A simple convolutional neural network (CNN) in PyTorch to classify images from the FashionMNIST dataset, with per-epoch training and test accuracy reporting.

## Features

- 3 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
- Adaptive device selection (CUDA, Apple MPS, or CPU)
- Per-epoch loss and accuracy on both training and test sets
- Configurable batch size, number of workers, and epochs via constants

## Requirements

- Python 3.9+  
- [PyTorch](https://pytorch.org)  
- torchvision  
- matplotlib  
- numpy  

All dependencies are listed in `requirements.txt`.

## Installation & Setup

1. **Clone the repo**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment**

   ```bash
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download the data & train**

   Simply run the training script—it will automatically download FashionMNIST:

   ```bash
   python train.py
   ```

   You’ll see per-epoch logs like:

   ```
   Epoch 0 Loss: 1234.56
   Epoch 0 Train Accuracy: 82.34% Test Accuracy: 80.12%
   …
   Finished Training
   ```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── train.py            # Main training + evaluation loop
└── data/               # Downloaded FashionMNIST files
```

## Configuration

Adjust these constants in `train.py` to tweak training:

- `BATCH_SIZE` (default: 128)  
- `NUM_WORKERS` (default: 0)  
- `EPOCHS` (default: 10)  

## License

This project is released under the MIT License. Feel free to reuse and adapt!