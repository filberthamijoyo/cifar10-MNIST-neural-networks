# Neural Network Implementation for CIFAR-10 and MNIST Datasets

This repository contains two distinct implementations of neural networks for image classification on the CIFAR-10 and MNIST datasets, demonstrating both PyTorch-based and custom implementations of deep learning techniques.

## Project Structure

- `custom_neural_network.ipynb`: Implementation focusing on CIFAR-10 with custom optimizations
  - Achieves 86.50% accuracy on CIFAR-10
  - Features enhanced data augmentation techniques
  - Implements custom training loops and optimizations
  
- `pytorch_implementation.ipynb`: Advanced implementation with custom neural network components
  - Achieves 93.26% accuracy on test set
  - Implements custom modules from scratch:
    - Softplus activation
    - Linear layer without bias
    - Cross-entropy loss
    - Custom optimizers (SGD)
  - Features MNIST dataset integration

## Features

### Neural Network Components
- Custom implementations of:
  - Module base class
  - Sequential container
  - Softplus activation function
  - Linear layer (without bias)
  - Cross-entropy loss
  - SGD optimizer

### Data Augmentation Techniques
- CIFAR-10 specific:
  - Random cropping (32x32 with padding=4)
  - Horizontal flipping
  - Color jittering (brightness=0.2, contrast=0.2, saturation=0.2)
  - Random rotation (15 degrees)
- Dataset-specific normalization
  - CIFAR-10: ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

### Training Optimizations
- Batch size: 256
- Number of epochs: 100
- GPU acceleration support
- Worker parallelization (num_workers=2)

## Datasets

### MNIST
- Handwritten digit recognition (0-9)
- 28x28 grayscale images
- 60,000 training images
- 10,000 test images
- Used for fundamental neural network understanding

### CIFAR-10
- 10 different classes of objects
- 32x32 color images
- 50,000 training images
- 10,000 test images
- More complex dataset for advanced techniques

## Results

The implementations achieve:
- CIFAR-10: 
  - Basic implementation: 86.50% accuracy
  - Advanced implementation: 93.26% accuracy
- MNIST: High accuracy with custom neural network components

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy
- cupy (for GPU acceleration)

## Usage

1. Install the required dependencies:
```bash
pip install torch torchvision matplotlib numpy cupy
```

2. Run the notebooks in your preferred Jupyter environment:
   - `custom_neural_network.ipynb` for CIFAR-10 focused implementation
   - `pytorch_implementation.ipynb` for custom neural network components and MNIST

## Implementation Details

### Custom Components
- Softplus activation: $y = \frac{1}{\beta} \ln(1+e^{\beta x})$
- Linear layer: $y = x W^T$
- Cross-entropy loss with numerical stability
- SGD optimizer with zero_grad functionality

### Training Features
- GPU acceleration when available
- Batch processing
- Enhanced data augmentation pipeline
- Custom training loops

## License

MIT License