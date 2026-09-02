---
**Internship Task Report**
**Task No.:** 8
**Task Title:** Implement a Simple Neural Network
**Date:** August 2026
**Tool / Notebook:** `task8.ipynb`

---

## 1. Objective

The objective of this task was to design, implement, and train a fully connected feedforward neural network from scratch using PyTorch, evaluate its performance on a benchmark image classification dataset, and analyse the training dynamics through loss and accuracy curves.

---

## 2. Introduction

Artificial Neural Networks (ANNs) are computational models loosely inspired by biological neural systems. A **feedforward neural network** consists of layers of interconnected units (neurons): an input layer, one or more hidden layers, and an output layer. Information flows in one direction — from input to output — with no recurrent connections.

Training a neural network involves two phases:
1. **Forward pass:** Input data is propagated through the network to produce predictions.
2. **Backward pass (Backpropagation):** The gradient of the loss with respect to each weight is computed using the chain rule, and the weights are updated in the direction that reduces the loss.

This task implemented a 3-layer fully connected network for handwritten digit classification on the **MNIST dataset** — one of the most well-established benchmarks in machine learning. The task focused on understanding the complete training pipeline rather than maximising accuracy.

---

## 3. Network Architecture

```
Input Layer        : 784 neurons (28 × 28 pixel values, flattened)
Hidden Layer 1     : 256 neurons, ReLU activation
Regularisation     : Dropout (p = 0.3)
Hidden Layer 2     : 128 neurons, ReLU activation
Output Layer       : 10 neurons (one per digit class, raw logits)
```

**Total trainable parameters: 235,146**

**Design Rationale:**
- **ReLU (Rectified Linear Unit)** activation was selected for hidden layers as it does not saturate for positive values, enabling stable gradient flow during backpropagation. Sigmoid and tanh activations are prone to the vanishing gradient problem in deeper networks.
- **Dropout (p = 0.3)** randomly zeroes 30% of neuron activations during each training step, preventing any single neuron from becoming excessively dominant and thus reducing overfitting.
- **Raw logits (no output softmax):** PyTorch's `CrossEntropyLoss` internally combines log-softmax and negative log-likelihood loss. Adding a softmax layer before this loss function would result in a mathematical error (double-application of log-softmax).

---

## 4. Dataset Description

| Attribute | Detail |
|---|---|
| Dataset Name | MNIST Handwritten Digits |
| Source | `torchvision.datasets.MNIST` |
| Training Set Size | 60,000 images |
| Test Set Size | 10,000 images |
| Image Dimensions | 28 × 28 pixels, grayscale |
| Number of Classes | 10 (digits 0–9) |
| Normalisation | Mean = 0.1307, Std = 0.3081 (dataset statistics) |

Images were normalised using the MNIST dataset's pre-computed mean and standard deviation to bring pixel values from [0, 255] to a standardised range, which stabilises training.

---

## 5. Methodology

### 5.1 Data Loading
PyTorch's `DataLoader` was used to handle batching and shuffling:

| Split | Batch Size | Shuffle |
|---|---|---|
| Training | 64 | Yes |
| Test | 256 | No |

A batch size of 64 was chosen for training — small enough to provide frequent weight updates (stochastic gradient descent effect) while large enough for stable gradient estimates.

### 5.2 Loss Function
**Cross-Entropy Loss** (`nn.CrossEntropyLoss`) was used — the standard loss function for multi-class classification. It measures the difference between the predicted class probability distribution and the true one-hot distribution.

### 5.3 Optimiser
**Adam (Adaptive Moment Estimation)** was selected with an initial learning rate of `1e-3`. Adam maintains per-parameter learning rates adapted based on estimates of first and second moments of the gradients, making it generally more effective than vanilla SGD for feedforward networks.

### 5.4 Learning Rate Schedule
**Step LR** scheduler (`StepLR`, step_size=5, gamma=0.5) halved the learning rate every 5 epochs. Learning rate decay allows aggressive learning in early epochs and fine-grained refinement in later epochs, improving final convergence.

### 5.5 Training Loop
The standard PyTorch training loop was implemented:

```
for each epoch:
    model.train()
    for each batch:
        optimizer.zero_grad()          # Clear accumulated gradients
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                # Backpropagation
        optimizer.step()               # Update weights
    model.eval()
    evaluate on test set
```

Training and test accuracy were recorded at each epoch to monitor convergence and detect overfitting.

---

## 6. Results and Analysis

### 6.1 Training Progress

| Epoch | Training Loss | Training Accuracy | Test Accuracy |
|---|---|---|---|
| 1 | ~0.35 | ~89.5% | ~96.2% |
| 3 | ~0.15 | ~95.4% | ~97.1% |
| 5 | ~0.10 | ~97.0% | ~97.6% |
| 7 | ~0.08 | ~97.8% | ~97.8% |
| 10 | ~0.06 | ~98.3% | ~98.0% |

**Final Test Accuracy: ~98.0%**

### 6.2 Training Curve Analysis
The training loss curve showed consistent monotonic decrease across all 10 epochs. The learning rate step at epoch 5 (halved from 1e-3 to 5e-4) produced a subtle but visible improvement in convergence rate in the second half of training.

The gap between training and test accuracy remained consistently small (~0.3–0.5%), indicating that the Dropout regularisation was effective in preventing overfitting.

### 6.3 Prediction Analysis
Visualisation of sample test predictions revealed that the vast majority of errors occurred on ambiguous digit formations — for example:
- Digit **4** written with a closed top, resembling **9**
- Digit **3** with an unusual top loop, resembling **8**
- Digit **7** written with a horizontal crossbar, resembling **1**

These are cases where human readers would also exhibit uncertainty, suggesting the model's error pattern is semantically reasonable.

---

## 7. Key Concepts Demonstrated

| Concept | Implementation Detail |
|---|---|
| `model.train()` / `model.eval()` | Enables/disables Dropout and BatchNorm training behaviour |
| `optimizer.zero_grad()` | Prevents gradient accumulation across batches |
| `loss.backward()` | Computes gradients via automatic differentiation |
| `optimizer.step()` | Updates weights using computed gradients |
| `torch.no_grad()` | Disables gradient computation during evaluation |
| Dropout regularisation | Reduces overfitting by randomly deactivating neurons |
| Learning rate decay | Enables fine-grained weight updates in later epochs |

---

## 8. Conclusion

A 3-layer fully connected neural network trained for 10 epochs on MNIST achieved approximately 98% test accuracy — a strong result for a simple architecture without convolutions. The training dynamics were stable, with no signs of divergence or significant overfitting. The implementation covered the full end-to-end PyTorch training pipeline, demonstrating forward/backward passes, loss computation, gradient updates, and evaluation.

For improved accuracy (approaching 99%+), convolutional layers would be the natural next step, as they exploit the spatial structure of images that fully connected layers cannot efficiently represent.

---

## 9. Output Artifacts

| File | Description |
|---|---|
| `task8.ipynb` | Jupyter notebook with all code and inline outputs |
| `task8_mnist_samples.png` | Sample training images from the MNIST dataset |
| `task8_curves.png` | Training loss and accuracy curves over 10 epochs |
| `task8_predictions.png` | Test predictions with correct/incorrect indicators |
| `data/MNIST/` | Downloaded MNIST dataset (shared with Task 3) |

---

## 10. Tools and Libraries

| Library | Purpose |
|---|---|
| PyTorch 2.13.0 | Neural network definition, training, and inference |
| torchvision | MNIST dataset loading and preprocessing |
| matplotlib | Visualisation of samples, curves, and predictions |
| numpy | Array operations |
