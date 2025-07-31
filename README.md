# Fashion MNIST Classification with PyTorch

A comprehensive PyTorch implementation for Fashion MNIST image classification with multiple training approaches, from basic to advanced architectures.

## 📋 Project Overview

This project demonstrates different approaches to training convolutional neural networks (CNNs) for the Fashion MNIST dataset, which contains 70,000 grayscale images of 10 different clothing categories.

### Dataset Classes
- T-shirt/top, Trouser, Pullover, Dress, Coat
- Sandal, Shirt, Sneaker, Bag, Ankle boot

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd torch_fashion_mnist

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training

Choose from three different training approaches:

#### Basic Training (Simple CNN)
```bash
python train.py --dataset fashionmnist --epochs 20 --batch-size 128
```

#### Manual Training (Custom Architecture)
```bash
python train_manual.py
```

#### Advanced Training (Complex Network)
```bash
python train_complex_network.py
```

## 📁 Project Structure

```
torch_fashion_mnist/
├── train.py                    # Modular training script with CLI arguments
├── train_manual.py            # Basic manual training implementation
├── train_complex_network.py   # Advanced network with optimizations
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── data/                     # Dataset storage (auto-downloaded)
├── best_fashion_model.pth    # Saved best model
└── .venv/                    # Virtual environment
```

## 🧠 Training Scripts

### 1. `train.py` - Modular Training Script
**Features:**
- Command-line interface with arguments
- Support for multiple datasets (Fashion MNIST, CIFAR-10)
- Comprehensive visualization (loss curves, confusion matrix)
- GPU/CPU/MPS device detection
- Performance metrics tracking

**Usage:**
```bash
python train.py --dataset fashionmnist --epochs 20 --batch-size 128 --lr 0.001 --save-plots
```

**Arguments:**
- `--dataset`: Dataset to use (`fashionmnist` or `cifar10`)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--save-plots`: Save training plots as images

### 2. `train_manual.py` - Basic Implementation
**Features:**
- Simple CNN architecture (3 conv layers + 2 FC layers)
- MaxPool2d for dimensionality reduction
- Basic training loop with accuracy tracking
- Educational implementation for learning PyTorch

**Usage:**
```bash
python train_manual.py
```

### 3. `train_complex_network.py` - Advanced Implementation
**Features:**
- Improved CNN architecture with BatchNorm and Dropout
- Data augmentation (random flips, rotations)
- Learning rate scheduling
- Model checkpointing
- Optimized for better performance

**Usage:**
```bash
python train_complex_network.py
```

## 🏗️ Model Architectures

### Basic Model (`train_manual.py`)
```
Input (1, 28, 28)
├── Conv2d(1→16, 3×3) + ReLU
├── MaxPool2d(2×2)
├── Conv2d(16→16, 3×3) + ReLU
├── MaxPool2d(2×2)
├── Conv2d(16→16, 3×3) + ReLU
├── Flatten
├── Linear(144→128) + ReLU
└── Linear(128→10)
```

### Advanced Model (`train_complex_network.py`)
```
Input (1, 28, 28)
├── Conv2d(1→32, 3×3, padding=1) + BatchNorm + ReLU + MaxPool
├── Conv2d(32→64, 3×3, padding=1) + BatchNorm + ReLU + MaxPool
├── Conv2d(64→128, 3×3, padding=1) + BatchNorm + ReLU + MaxPool
├── AdaptiveAvgPool2d(1×1)
├── Flatten
├── Linear(128→256) + ReLU + Dropout
├── Linear(256→128) + ReLU + Dropout
└── Linear(128→10)
```

## ⚙️ Requirements

```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.60.0
numpy>=1.21.0
```

## 🔧 Device Support

The project automatically detects and uses the best available device:

- **CUDA GPU**: NVIDIA GPUs on Windows/Linux
- **MPS**: Apple Silicon (M1/M2) on macOS
- **CPU**: Fallback for all other systems

## 📊 Performance Metrics

### Expected Results
- **Basic Model**: ~85-90% accuracy
- **Advanced Model**: ~92-95% accuracy

### Training Time (approximate)
- **CPU**: 10-15 minutes for 20 epochs
- **GPU**: 2-5 minutes for 20 epochs

## 🎯 Key Features

### Training Optimizations
- **Batch Normalization**: Stabilizes training and speeds convergence
- **Dropout**: Prevents overfitting
- **Data Augmentation**: Improves generalization
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Model Checkpointing**: Saves best performing model

### Visualization
- Training/validation loss curves
- Accuracy progression
- Confusion matrix
- Sample predictions with confidence scores

### Monitoring
- Real-time accuracy tracking
- Loss monitoring
- Device utilization display
- Progress bars with tqdm

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train.py --batch-size 64
```

**2. Slow Training**
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"
```

**3. Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Performance Tips

1. **Use GPU**: Training is significantly faster on GPU
2. **Adjust Batch Size**: Larger batches for GPU, smaller for CPU
3. **Reduce Workers**: Lower `num_workers` if experiencing high CPU usage
4. **Monitor Memory**: Use `nvidia-smi` (GPU) or `htop` (CPU) to monitor usage

## 📈 Extending the Project

### Adding New Datasets
1. Add dataset loading logic in `get_dataloaders()`
2. Update model architecture for new input dimensions
3. Add dataset choice to argument parser

### Custom Model Architecture
1. Create new model class inheriting from `nn.Module`
2. Define forward pass with your architecture
3. Update model instantiation in training script

### Hyperparameter Tuning
- Learning rate: Try 0.001, 0.0001, 0.01
- Batch size: 32, 64, 128, 256
- Dropout rate: 0.1, 0.3, 0.5
- Number of epochs: 10, 20, 50

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Happy Training! 🚀**