# Brain Tumor MRI Classification

## Project Overview
This project implements a deep learning system for classifying brain MRI images into four categories. Using transfer learning with the **Xception** architecture pre-trained on ImageNet, combined with custom fully connected layers and dropout regularization, the model achieves high accuracy in detecting and classifying brain tumors. The project demonstrates the power of Convolutional Neural Networks (CNNs) in medical imaging applications.

---

## Dataset
- MRI images organized into:
  - `Training` folder (images for training)
  - `Testing` folder (images for evaluation)
- Dataset includes four tumor categories.
- Data split into:
  - Training set
  - Validation set
  - Test set
- Stratified splitting ensures balanced class distribution across all sets.

---

## Features
- Transfer learning with **Xception** pre-trained on ImageNet
- Image preprocessing:
  - Rescaling pixels to [0,1]
  - Brightness augmentation for training
- Custom classifier layers:
  - Flatten → Dense → Dropout → Dense → Softmax
- Metrics tracked:
  - Accuracy
  - Precision
  - Recall
- Evaluation:
  - Confusion matrix
  - Classification report per class
- Optimizer: **Adamax**
- Loss function: **Categorical Crossentropy**
- Batch size: 32
- Image size: 299x299

---

## Installation

```bash
pip install tensorflow numpy pandas scikit-learn pillow matplotlib
````

---

## Usage

1. Prepare your dataset in the following structure:

```
/Training
    /Class1
    /Class2
    /Class3
    /Class4
/Testing
    /Class1
    /Class2
    /Class3
    /Class4
```

2. Update paths in the code:

```python
train_path = '/path/to/Training'
test_path = '/path/to/Testing'
```

3. Run the Python script or notebook:

```python
python train_brain_tumor_model.py
```

4. Evaluate the model:

* Check **train, validation, and test accuracy**
* Generate **classification report**
* Generate **confusion matrix** for visual analysis

---

## Model Architecture

* **Base model:** Xception (pretrained, include_top=False, pooling='max')
* **Custom layers:**

  * Flatten
  * Dropout(0.3)
  * Dense(128, ReLU)
  * Dropout(0.25)
  * Dense(4, Softmax)
* Transfer learning enabled: leveraging pre-trained features
* Optional fine-tuning for top layers to further improve performance

---

## Training

* Data generators:

  * `ImageDataGenerator` for training with augmentation
  * Validation and test generators without augmentation
* Training parameters:

  * Epochs: 10
  * Batch size: 32
  * Optimizer: Adamax
  * Learning rate: 0.001
* Metrics monitored:

  * Accuracy
  * Precision
  * Recall

---

## Evaluation & Results

* High accuracy on training, validation, and test sets
* Detailed per-class metrics (precision, recall, F1-score)
* Confusion matrix for error analysis

Example:

```
Train Accuracy: 98.5%
Validation Accuracy: 96.7%
Test Accuracy: 95.8%
```
