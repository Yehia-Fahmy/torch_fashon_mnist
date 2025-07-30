# Multi-Classification PyTorch Project

This project provides a modular framework for training multi-class image classifiers using PyTorch, with GPU support, dataset-agnostic loading, and comprehensive performance tracking and visualization.

## Features
- Supports multiple datasets (Fashion MNIST, CIFAR-10, easily extensible)
- Automatic GPU/CPU selection
- Modular model and data loader
- Tracks loss, accuracy, and per-class metrics
- Visualizes training/validation curves, confusion matrix, and sample predictions

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training:
   ```bash
   python train.py --dataset cifar10 --epochs 20 --batch-size 128
   ```
   (Use `--dataset fashionmnist` for Fashion MNIST)

## Customization
- Add new datasets or models by editing `train.py` (or modular files if split).

## Visualization
- After training, loss/accuracy curves and confusion matrix will be saved as images.

## Python Virtual Environment Setup

It is recommended to use a Python virtual environment to manage dependencies and avoid conflicts with other projects.

### Create and Activate a Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

After activating the environment, install the dependencies as shown below:

```bash
pip install -r requirements.txt
```

---

### 1. **Check Device Selection in Code**
Your code uses:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
This will use the GPU if PyTorch detects CUDA. However, on macOS, CUDA is not available (NVIDIA GPUs only), and Apple Silicon (M1/M2) uses `mps` for GPU acceleration.

### 2. **Update Device Selection for macOS (Apple Silicon)**
To support Apple Silicon, update your device selection logic to:
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```
This will use:
- CUDA GPU if available (Linux/Windows with NVIDIA)
- MPS (Metal Performance Shaders) if available (macOS with Apple Silicon)
- CPU otherwise

### 3. **Verify Device Usage in Training**
Add a print statement at the start of your script:
```python
print(f"Using device: {device}")
```
This will confirm which device is being used.

---

#### Would you like me to update your `train.py` with this improved device selection and a device printout?