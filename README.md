# Neural Network Implementation for CIFAR-10 and MNIST Datasets

This repository contains two distinct implementations of neural networks for image classification:
1. A neural network built entirely from scratch using only NumPy/CuPy for CIFAR-10
2. A custom PyTorch-based implementation with advanced components for both CIFAR-10 and MNIST

## Project Structure

- `custom_neural_network.ipynb`: Pure NumPy/CuPy implementation built from scratch
  - No deep learning frameworks used - implemented using only numerical computing libraries
  - Complete implementation of forward and backward passes
  - Custom gradient computation and optimization
  - Achieves 86.50% accuracy on CIFAR-10
  - Features enhanced data augmentation techniques
  - Demonstrates deep understanding of neural network fundamentals
  
- `pytorch_implementation.ipynb`: Advanced implementation with custom PyTorch components
  - Achieves 93.26% accuracy on CIFAR-10 test set
  - Implements custom modules from scratch within PyTorch framework:
    - Softplus activation with numerical stability
    - Linear layer without bias for efficient computation
    - Cross-entropy loss with improved stability
    - Custom optimizers (SGD, Adam)
  - Features MNIST dataset integration with 97.59% accuracy

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
    - Learning rate: 0.01 with momentum
  - MNIST:
    - Batch size: 128
    - Number of epochs: 5
    - Learning rate: 1e-4 with Adam optimizer
- Hardware utilization:
  - GPU acceleration with CUDA support
  - Worker parallelization (num_workers=2)
  - Memory-efficient batch processing

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
  - Includes threshold for numerical stability
- Linear layer: $y = x W^T$
  - Efficient matrix operations
  - Optional bias term
- Cross-entropy loss with numerical stability:
  - Softmax with temperature scaling
  - Log-sum-exp trick for numerical stability
- Optimizers:
  - SGD with momentum: $v = \gamma v + \eta \nabla f(x)$
  - Adam: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$

### Training Features
- GPU acceleration when available
- Efficient batch processing
- Enhanced data augmentation pipeline
- Custom training loops with progress tracking
- Early stopping capability
- Learning rate scheduling support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you find this implementation useful in your research, please consider citing:

```bibtex
@misc{hamijoyo2024neural,
  author = {Filbert Hamijoyo},
  title = {Neural Network Implementation for CIFAR-10 and MNIST},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/filberthamijoyo/cifar10-MNIST-neural-networks}
}
```

## Let's Connect

I'm open to:
- Research collaborations in deep learning and computer vision
- Discussions about neural network implementations
- Optimization techniques for deep learning models
- Knowledge sharing and mentorship

Feel free to reach out for collaboration or discussion!