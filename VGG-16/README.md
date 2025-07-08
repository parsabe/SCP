# VGG-16 Model Documentation

## Overview

This documentation covers the implementation details, design choices, and experimental setup of the **VGG-16-based Image Classification Pipeline** developed as part of the **Scientific Computing Project (SCP)**. The VGG-16 architecture serves as the **baseline model** in this project and will be followed by additional models in the future.

---

## Model Architecture

The VGG-16 model in this project has been implemented with the following characteristics:

- **Base Model:** Classical VGG-16 structure with modifications
- **Modifications:**
  - Additional convolutional layers
  - Batch Normalization after each convolution
  - Light Dropout regularization
  - Optional SE-Blocks and DropBlock regularizers

The model is initialized using PyTorch and extended for flexibility in research experimentation.

---

Features
--------

- **Research-Grade Modular Design**
  - Multiple architecture support (VGG-16, others upcoming)
  - Clear folder structure

- **Training Strategy:**
  - Hyperparameter Tuning with K-Fold Cross-Validation (K=3)
  - Optimizer comparison (AdamW & SGD with Momentum)
  - Early stopping & learning rate scheduling

- **Data Augmentation:**
  - Horizontal/Vertical Flip
  - Random Rotation, Shift, Scale
  - Color Jitter
  - CutMix / MixUp (optional)

- **Evaluation Metrics & Visualization:**
  - Loss and Accuracy Curves
  - ROC & PR Curves
  - Confusion Matrix
  - F1 Score & Per-Class Accuracy
  - Hyperparameter Tuning Result Visualization
  - t-SNE Embedding Visualization
---

## Hyperparameter Tuning Configuration

The training process includes an automatic hyperparameter tuning phase to improve model generalization and performance.

**Tuning Setup:**

- **K-Folds Cross-Validation:** 3
- **Number of Epochs:** 50
- **Early Stopping Patience:** 7
- **Learning Rate Grid:** `[0.001, 0.0005, 0.0001]`
- **Batch Size Grid:** `[32, 64, 128]`
- **Optimizers:** `AdamW` and `SGD with Momentum`

**Tuning Metric:** Validation Accuracy (mean across K folds)

---

## 🧩 Data Augmentation & Preparation

To improve generalization and mitigate overfitting, various data augmentation techniques are applied:

- Random Horizontal & Vertical Flips
- Random Rotation, Shift, and Scaling
- Color Jitter
- CutMix / MixUp (optional)

The dataset is split into **Train**, **Validation**, and **Test** sets.

---

## 🔥 Training Pipeline

The training process is composed of the following steps:

1. **Data Loading & Augmentation:** Datasets are loaded, split, and augmented.
2. **Model Initialization:** VGG-16 model initialized with modified architecture.
3. **Hyperparameter Tuning:** Grid Search over defined parameters using Cross-Validation.
4. **Training:**
   - Early stopping based on validation loss.
   - Learning Rate Scheduling.
   - Logging of training and validation metrics.
5. **Evaluation:**
   - Test accuracy calculated.
   - Metrics and plots saved in `results/`.

---

## 📊 Evaluation Metrics & Visualization

The following evaluation outputs are generated and saved:

- Training & Validation Loss Curves
- Training & Validation Accuracy Curves
- Learning Rate Schedules
- Confusion Matrix
- Per-Class Accuracy Bar Plot
- ROC & PR Curves per class
- F1 Score per class
- t-SNE Embedding Visualization
- Hyperparameter Grid Result Visualization
- Optimizer Comparison Charts (AdamW vs SGD)
- Error analysis: Misclassification samples, lowest confidence samples
- Calibration Curve

The pipeline offers:

- Full control over training experiments
- K-Fold Cross-Validation based hyperparameter tuning
- Strong data augmentation & evaluation methodology
- Clear and clean structure for further model extensions

---

Benchmarks
----------

The evaluation results and visualizations are saved under `results/` directory. They include:

| Metric                      | Description                           |
|:----------------------------|:--------------------------------------|
| Training Loss Curve        | Loss vs Epochs during training        |
| Validation Accuracy Curve  | Accuracy vs Epochs on validation set  |
| Confusion Matrix           | Per-class classification visualization|
| ROC & PR Curves            | Receiver Operating & Precision-Recall |
| F1 Score                   | Per-class evaluation metric           |
| t-SNE Plot                 | Embedding visualization               |
| Hyperparameter Results     | Grid search visualization             |
| Optimizer Comparison       | AdamW vs SGD test accuracy            |
---

## 🔮 Future Extensions

The current pipeline is designed to support further extensions:

- Additional model architectures (ResNet, DenseNet, ViT)
- Advanced optimizer integrations (Lion, RMSprop)
- Grad-CAM explainability
- REST API deployment
- Experiment Tracking integration

---

## 👥 Contributors

- <a href="https://github.com/hounaar">Parsa Besharat</a>
- <a href="https://github.com/Lars314159">Lars Wunderlich</a>
- <a href="https://github.com/ToniMahojoni">Toni Sand</a>

---

## 📄 License

This project is open-source and available under the **MIT License**.

