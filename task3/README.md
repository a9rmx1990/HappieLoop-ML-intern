---
**Internship Task Report**
**Task No.:** 3
**Task Title:** Image Classification with a Pre-trained Model
**Date:** August 2026
**Tool / Notebook:** `task3.ipynb`

---

## 1. Objective

The objective of this task was to demonstrate image classification using a pre-trained deep learning model, without training any model from scratch. The task covers model loading, image preprocessing, inference execution, top-k prediction decoding, and visualisation of results, while also exploring the behaviour of pre-trained classifiers on out-of-distribution inputs.

---

## 2. Introduction

Pre-trained deep learning models represent one of the most practically significant developments in modern AI. Instead of requiring millions of images and days of compute time to train a model from scratch, practitioners can download a model that has already been trained on a large benchmark dataset and apply it directly — or fine-tune it — on a new task. This paradigm is known as **transfer learning**.

In this task, **MobileNetV2** — a lightweight convolutional neural network pre-trained on the ImageNet-1k benchmark — was used to classify images. The model was not retrained; only its inference capability was exercised.

The test images used were MNIST handwritten digits, sourced from the local disk (already downloaded during Task 8). This deliberately introduces an **out-of-distribution scenario**: the model was trained on natural photographs but is tested on handwritten digit scans. This choice was made to highlight an important and often overlooked property of softmax-based classifiers — their tendency to produce overconfident predictions even when the input is completely unfamiliar.

---

## 3. Model Description

| Attribute | Detail |
|---|---|
| Model Architecture | MobileNetV2 |
| Pre-trained On | ImageNet-1k (1,281,167 training images, 1000 classes) |
| Source | `torchvision.models.mobilenet_v2` |
| Framework | PyTorch 2.13.0 |
| Inference Device | CPU |
| Total Parameters | ~3.4 million |
| Input Dimensions | 3 × 224 × 224 (RGB) |
| Top-1 Accuracy (ImageNet) | 71.8% |

MobileNetV2 was chosen for its compact architecture, fast CPU inference speed, and wide adoption as a benchmark mobile model. It uses inverted residuals and linear bottlenecks to achieve competitive accuracy with far fewer parameters than heavier models such as VGG or ResNet.

---

## 4. Dataset Description (Test Images)

| Attribute | Detail |
|---|---|
| Dataset | MNIST Handwritten Digits |
| Source | `torchvision.datasets.MNIST` (pre-downloaded) |
| Images Used | 10 (one per digit class: 0–9) |
| Image Format | 28×28 grayscale |
| Distribution Status | Out-of-distribution relative to ImageNet |

Note: MNIST images were converted from grayscale to RGB by replicating the single channel three times — a standard approach required when feeding single-channel images to RGB-trained models.

---

## 5. Methodology

### 5.1 Model Loading
The model was loaded with official pretrained weights using the modern torchvision API:

```python
weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)
model.eval()
```

Setting `model.eval()` is mandatory for inference — it disables Dropout layers and switches BatchNorm layers to use running statistics instead of batch statistics.

### 5.2 Class Labels
ImageNet class names were retrieved directly from the weights metadata object:

```python
imagenet_labels = weights.meta['categories']
```

This approach is preferred over downloading a separate labels file, as it guarantees label-to-index alignment with the specific model version being used.

### 5.3 Preprocessing Pipeline
The recommended preprocessing transform was obtained from the weights metadata:

```python
preprocess = weights.transforms()
```

This automatically applies: resize to 256 → centre crop to 224 → convert to tensor → normalise with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

### 5.4 Inference
For each test image, inference was executed inside a `torch.no_grad()` context (disabling gradient computation) and the output logits were converted to probabilities via softmax. The top-5 predictions were extracted using `torch.topk`.

```python
with torch.no_grad():
    logits = model(tensor)
probs = torch.nn.functional.softmax(logits[0], dim=0)
top5_probs, top5_idx = torch.topk(probs, 5)
```

### 5.5 Visualisations
Three visualisations were produced:
1. **Prediction Grid:** 2×5 grid of MNIST digit images annotated with the top-1 ImageNet prediction and confidence.
2. **Top-5 Bar Charts:** Horizontal bar charts showing the top-5 class probabilities for each digit.
3. **Confidence Bar Chart:** Comparison of top-1 confidence levels across all 10 digit classes.

---

## 6. Results and Analysis

### 6.1 Sample Predictions

The model assigned ImageNet labels that reflect visual similarity to the digit shapes rather than digit identity. Selected examples:

| MNIST Digit | Top-1 Prediction | Confidence | Observation |
|---|---|---|---|
| 0 | dial telephone / sundial | ~40–60% | Circular shape matches round objects |
| 1 | ballpoint pen / ruler | ~30–50% | Thin vertical stroke resembles elongated objects |
| 8 | chain / pretzel | ~20–40% | Looped shape resembles connected structures |

### 6.2 Softmax Overconfidence

A critical observation from this experiment is that the model produces **high-confidence predictions** even on inputs that are completely outside its training distribution. This is a direct consequence of the softmax activation function — because probabilities must sum to 1, the model is forced to assign most of its probability mass to some class, regardless of whether the input makes sense.

This phenomenon is well-documented in literature and represents a significant practical risk in deployed systems. A classifier may assign 80% confidence to an incorrect class when shown a completely unfamiliar image. Without a separate **out-of-distribution (OOD) detection** mechanism, this behaviour can cause silent, hard-to-detect failures in production.

---

## 7. Key Concepts Demonstrated

| Concept | Description |
|---|---|
| Transfer Learning | Using a model trained on one dataset (ImageNet) directly for inference |
| `model.eval()` | Disables training-specific layers for correct inference behaviour |
| `torch.no_grad()` | Reduces memory usage and speeds up inference by disabling gradient tracking |
| Preprocessing alignment | Input normalisation must exactly match training-time normalisation |
| Softmax overconfidence | Softmax models produce high-confidence predictions even on OOD inputs |
| Out-of-distribution inputs | Inputs that do not match the model's training data distribution |

---

## 8. Conclusion

This task demonstrated the complete inference pipeline for a pre-trained image classification model. MobileNetV2 was successfully loaded, applied, and its outputs decoded without any model training. The deliberate use of out-of-distribution MNIST images surfaced an important behaviour of softmax classifiers — overconfident predictions on unfamiliar inputs — which is a critical consideration when deploying such models in real-world applications.

For production use cases, OOD detection methods such as temperature scaling, energy-based scoring, or dedicated OOD classifiers should be applied alongside the primary classifier to flag uncertain or unfamiliar inputs.

---

## 9. Output Artifacts

| File | Description |
|---|---|
| `task3.ipynb` | Jupyter notebook with all code and inline outputs |
| `task3_mnist_predictions.png` | 2×5 grid of digits with top-1 ImageNet predictions |
| `task3_top5.png` | Top-5 confidence bar charts per digit |
| `task3_confidence.png` | Top-1 confidence comparison across digit classes |
| `imagenet_labels.json` | Full list of 1000 ImageNet class names |
| `data/` | Symlink to shared MNIST data directory (from Task 8) |

---

## 10. Tools and Libraries

| Library | Purpose |
|---|---|
| PyTorch 2.13.0 | Model loading, inference, tensor operations |
| torchvision | Pre-trained model weights, transforms, MNIST dataset |
| PIL (Pillow) | Image mode conversion (grayscale → RGB) |
| matplotlib | Visualisation |
| numpy | Array operations |
