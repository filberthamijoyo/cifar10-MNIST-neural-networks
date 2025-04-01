# Neural Network Implementation for CIFAR-10 and MNIST Datasets

A deep learning project featuring two distinct neural network implementations for image classification:

1. A pure NumPy/CuPy convolutional neural network implementation for CIFAR-10 (86.50% accuracy)
   - Built entirely from scratch with no deep learning frameworks
   - Custom implementation of convolution operations, backpropagation, and optimization
   - Demonstrates fundamental neural network concepts and mathematics

2. An advanced PyTorch implementation with custom components (93.26% CIFAR-10, 97.59% MNIST)
   - Custom modules: Softplus activation, efficient Linear layer, stable Cross-entropy
   - ResNet-inspired architecture with skip connections and BatchNorm
   - Advanced training features: mixed precision, CuDNN optimization, custom schedulers

Both implementations feature comprehensive data augmentation pipelines, GPU acceleration, and detailed hyperparameter configurations.

## Project Structure

- `custom_neural_network.ipynb`: Pure NumPy/CuPy implementation built from scratch
  - No deep learning frameworks used - implemented using only numerical computing libraries
  - Complete implementation of forward and backward passes
  - Custom gradient computation and optimization
  - Achieves 86.50% accuracy on CIFAR-10
  - Features enhanced data augmentation techniques
  - Demonstrates deep understanding of neural network fundamentals
  - Architecture:
    - 3 Convolutional layers (32, 64, 128 filters)
    - 2 Max pooling layers
    - 2 Fully connected layers (512, 10 units)
    - ReLU activation functions
    - Softmax output layer
  
- `pytorch_implementation.ipynb`: Advanced implementation with custom PyTorch components
  - Achieves 93.26% accuracy on CIFAR-10 test set
  - Achieves 97.59% accuracy on MNIST test set
  - Implements custom modules from scratch within PyTorch framework:
    - Softplus activation with numerical stability
    - Linear layer without bias for efficient computation
    - Cross-entropy loss with improved stability
    - Custom optimizers (SGD, Adam)
  - Architecture:
    - ResNet-like structure with skip connections
    - Batch Normalization after each convolution
    - Dropout (p=0.5) for regularization
    - LeakyReLU activations (negative_slope=0.01)

## Features

### From-Scratch Neural Network Components (custom_neural_network.ipynb)
- Pure NumPy/CuPy implementation of:
  - Convolutional layers with custom kernel operations
  - Backpropagation algorithm
  - Gradient descent optimization
  - Loss function computation
  - Forward and backward passes
  - Batch processing mechanisms

### Advanced PyTorch Components (pytorch_implementation.ipynb)
- Custom implementations of:
  - Module base class with automatic gradient tracking
  - Sequential container for layer organization
  - Softplus activation function with stability improvements
  - Linear layer (without bias) for efficient computation
  - Cross-entropy loss with numerical stability
  - SGD and Adam optimizers with momentum

### Data Augmentation Techniques
- CIFAR-10 specific:
  - Random cropping (32x32 with padding=4)
  - Horizontal flipping for dataset augmentation
  - Color jittering (brightness=0.2, contrast=0.2, saturation=0.2)
  - Random rotation (15 degrees)
- Dataset-specific normalization
  - CIFAR-10: ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
  - MNIST: ((0.1307,), (0.3081,))

### Training Optimizations
- Hyperparameter configurations:
  - CIFAR-10: 
    - Batch size: 256
    - Number of epochs: 100
    - Learning rate: 0.01 with momentum (0.9)
    - Weight decay: 5e-4
    - Learning rate schedule: reduce by 0.1 at epochs [50, 75]
  - MNIST:
    - Batch size: 128
    - Number of epochs: 5
    - Learning rate: 1e-4 with Adam optimizer
    - Betas: (0.9, 0.999)
    - Epsilon: 1e-8
- Hardware utilization:
  - GPU acceleration with CUDA support
  - Worker parallelization (num_workers=2)
  - Memory-efficient batch processing
  - CuDNN benchmarking enabled
  - Mixed precision training (FP16)

## Datasets

### MNIST
- Handwritten digit recognition (0-9)
- Dataset characteristics:
  - 28x28 grayscale images
  - 60,000 training images
  - 10,000 test images
  - 10 classes (digits 0-9)
- Used for fundamental neural network understanding
- Normalized with mean=0.1307, std=0.3081

### CIFAR-10
- Natural image classification
- Dataset characteristics:
  - 32x32 RGB color images
  - 50,000 training images
  - 10,000 test images
  - 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- More complex dataset for advanced techniques
- Per-channel normalization

## Results

The implementations achieve:
- CIFAR-10: 
  - From-scratch implementation (custom_neural_network.ipynb): 86.50% accuracy
    - Notable achievement for a pure NumPy/CuPy implementation
    - Demonstrates effectiveness of custom gradient computation
  - Advanced implementation (pytorch_implementation.ipynb): 93.26% accuracy
    - Leverages custom PyTorch components
    - Benefits from advanced optimization techniques
- MNIST: 97.59% accuracy on the test set
  - Custom implementation with CNN architecture
  - Using BatchNorm, LeakyReLU, and Dropout for regularization
  - Trained with Adam optimizer (β1=0.9, β2=0.999, ε=1e-8)

## Requirements

- Python 3.x
- Core dependencies:
  - numpy>=1.19.0
  - cupy>=9.0.0 (for GPU acceleration)
  - PyTorch>=1.7.0
  - torchvision>=0.8.0
- Visualization and utilities:
  - matplotlib>=3.3.0
  - jupyter>=1.0.0
  - pillow>=8.0.0

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install the required dependencies:
```bash
pip install torch torchvision matplotlib numpy cupy pillow jupyter
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/filberthamijoyo/cifar10-MNIST-neural-networks.git
cd cifar10-MNIST-neural-networks
```

2. Run the notebooks in your preferred Jupyter environment:
   - `custom_neural_network.ipynb` for the from-scratch CIFAR-10 implementation
   - `pytorch_implementation.ipynb` for custom PyTorch components and MNIST

## Implementation Details

### Custom Components
- Softplus activation: $y = \frac{1}{\beta} \ln(1+e^{\beta x})$
  - Includes threshold for numerical stability (β=1)
  - Smooth alternative to ReLU
- Linear layer: $y = x W^T$
  - Efficient matrix operations using CUDA
  - Optional bias term
  - Xavier/Glorot initialization
- Cross-entropy loss with numerical stability:
  - Softmax with temperature scaling (τ=1)
  - Log-sum-exp trick for numerical stability
  - Label smoothing (ε=0.1)
- Optimizers:
  - SGD with momentum: $v = \gamma v + \eta \nabla f(x)$
  - Adam: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
    - Bias correction for first and second moments
    - AMSGrad variant for better convergence

### Training Features
- GPU acceleration when available
- Efficient batch processing
- Enhanced data augmentation pipeline
  - Random crop with reflection padding
  - Random horizontal flip (p=0.5)
  - Color jittering with random brightness, contrast, and saturation
  - Cutout regularization (16x16 patches)
- Custom training loops with progress tracking
- Early stopping with patience=10
- Learning rate scheduling:
  - Step decay
  - Cosine annealing
  - Warm-up phase (5 epochs)

## License

MIT License