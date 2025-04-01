# Neural Network Implementation for CIFAR-10 and MNIST Datasets

This repository contains PyTorch implementations of neural networks for image classification on both the CIFAR-10 and MNIST datasets. The project demonstrates various deep learning techniques and optimizations to achieve high accuracy on these fundamental computer vision tasks.

## Project Structure

- `custom_neural_network.ipynb`: Implementation of neural networks from scratch with custom optimizations for both MNIST and CIFAR-10
- `pytorch_implementation.ipynb`: PyTorch-based implementation with advanced techniques
- `technical_report.pdf`: Detailed analysis and results of the implementations

## Datasets

### MNIST
- Handwritten digit recognition (0-9)
- 28x28 grayscale images
- 60,000 training images
- 10,000 test images

### CIFAR-10
- 10 different classes of objects
- 32x32 color images
- 50,000 training images
- 10,000 test images

## Features

- Custom neural network implementations for both datasets
- Data augmentation techniques including:
  - Random cropping
  - Horizontal flipping (for CIFAR-10)
  - Color jittering (for CIFAR-10)
  - Rotation
- Advanced optimization methods
- Performance analysis and comparisons
- Dataset-specific optimizations

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

## Results

The implementations achieve:
- Over 86.50% accuracy on the CIFAR-10 test set
- High accuracy on the MNIST dataset
demonstrating effective learning and generalization across different types of image classification tasks.

## Usage

1. Install the required dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

2. Run the notebooks in your preferred Jupyter environment

## License

MIT License 