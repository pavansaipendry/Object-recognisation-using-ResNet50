# Object-recognition-using-ResNet50 

This project focuses on image classification of the CIFAR-10 dataset, utilizing both a simple neural network and a deep learning model with transfer learning using ResNet50. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Neural Network Model](#neural-network-model)
- [Transfer Learning with ResNet50](#transfer-learning-with-resnet50)
- [Results](#results)

## Dataset
The dataset used is the [CIFAR-10](https://www.kaggle.com/c/cifar-10) dataset, which is a popular dataset for image classification tasks.

- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Number of images**: 60,000 (50,000 training images and 10,000 test images)
- **Image size**: 32x32 pixels with 3 color channels (RGB)

## Project Structure
The project is divided into two main parts:
1. A simple feed-forward neural network model trained from scratch.
2. Transfer learning using ResNet50, a pre-trained deep learning model, for improved performance.

## Neural Network Model
We built a simple neural network model using TensorFlow and Keras. The model consists of:
- An input layer to flatten the 32x32x3 input image.
- Two dense layers: one with 64 neurons and ReLU activation, and the output layer with 10 neurons and softmax activation.

### Training Details
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical cross-entropy
- **Epochs**: 10
- **Validation Split**: 0.1

### Epoch Results
| Epoch | Train Accuracy | Train Loss | Validation Accuracy | Validation Loss |
|-------|----------------|------------|---------------------|-----------------|
| 1     | 24.3%          | 2.0656     | 29.2%               | 1.8968          |
| 2     | 33.5%          | 1.8564     | 34.7%               | 1.8159          |
| 3     | 35.5%          | 1.8029     | 33.4%               | 1.8248          |
| 4     | 36.0%          | 1.7821     | 35.2%               | 1.7887          |
| 5     | 37.8%          | 1.7439     | 36.2%               | 1.7618          |
| 6     | 37.7%          | 1.7427     | 37.2%               | 1.7521          |
| 7     | 38.4%          | 1.7265     | 35.9%               | 1.7660          |
| 8     | 38.6%          | 1.7148     | 36.1%               | 1.7653          |
| 9     | 38.4%          | 1.7114     | 38.4%               | 1.7323          |
| 10    | 39.3%          | 1.7099     | 35.9%               | 1.7508          |

## Transfer Learning with ResNet50
To achieve better accuracy, we employed transfer learning using ResNet50, a pre-trained model on the ImageNet dataset. ResNet50 was modified to fit our classification problem.

### Model Details
- The model first resizes the input images to match the input requirements of ResNet50.
- We then added several dense layers with dropout and batch normalization to enhance performance.

### Training Details
- **Optimizer**: RMSprop with a learning rate of `2e-5`
- **Loss Function**: Sparse categorical cross-entropy
- **Epochs**: 10
- **Validation Split**: 0.1

### Epoch Results
| Epoch | Train Accuracy | Train Loss | Validation Accuracy | Validation Loss |
|-------|----------------|------------|---------------------|-----------------|
| 1     | 33.6%          | 2.0165     | 78.0%               | 0.7805          |
| 2     | 68.5%          | 1.0239     | 88.4%               | 0.4734          |
| 3     | 80.9%          | 0.7190     | 91.9%               | 0.3342          |
| 4     | 87.4%          | 0.5447     | 93.2%               | 0.2724          |
| 5     | 91.5%          | 0.4166     | 93.4%               | 0.2547          |
| 6     | 94.2%          | 0.3205     | 93.4%               | 0.2383          |
| 7     | 95.5%          | 0.2640     | 93.9%               | 0.2135          |
| 8     | 96.6%          | 0.2138     | 94.4%               | 0.2058          |
| 9     | 97.5%          | 0.1731     | 93.8%               | 0.2291          |
| 10    | 97.9%          | 0.1461     | 93.9%               | 0.2234          |

## Results
- The simple neural network achieved a maximum training accuracy of 39.3% and validation accuracy of 38.4%.
- The ResNet50 model using transfer learning showed significant improvement, achieving a maximum training accuracy of 97.9% and validation accuracy of 94.4%.
