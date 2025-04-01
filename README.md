# VGG-16 Image Classification Project

A fully customized and optimized **VGG-16-based deep learning pipeline** for image classification, designed to maximize model performance, reduce underfitting, and systematically compare optimizers (AdamW & SGD) using structured hyperparameter tuning and evaluation.

---

## 🟢 Key Features & Improvements

- **Modified VGG-16 Architecture:**
  - Additional convolutional layers for deeper feature extraction.
  - Batch Normalization after every Conv layer.
  - Light Dropout regularization.
  - Optional SE-Blocks and DropBlock modules.

- **Training Strategy:**
  - **Hyperparameter Tuning** with K-Fold Cross-Validation (K=3).
  - Optimizer comparison: **AdamW** vs **SGD with Momentum**.
  - Exponential learning rate decay.
  - Early stopping with patience=7.

- **Data Augmentation:**
  - Random Horizontal & Vertical Flip
  - Random Rotation, Shift, Scale
  - Color Jitter
  - CutMix / MixUp (optional)

- **Loss Function:**
  - Label Smoothing Cross Entropy Loss

- **Evaluation & Visualization:**
  - Training & Validation Loss/Accuracy curves
  - Learning Rate Schedules
  - Confusion Matrix & Classification Report
  - ROC & PR Curves
  - F1 Score per class
  - t-SNE embedding visualization
  - Error analysis (lowest confidence, highest misclassifications)
  - Hyperparameter Tuning Result Visualization

---

## ⚙️ Hyperparameter Tuning Configuration

- **K-FOLDS:** 3
- **NUM_EPOCHS:** 50
- **PATIENCE:** 7
- **Learning Rate Grid:** `[0.001, 0.0005, 0.0001]`
- **Batch Size Grid:** `[32, 64, 128]`
- **Optimizers:** `AdamW` and `SGD with Momentum`

**Tuning Metric:** Validation Accuracy (average across folds)

---

## 🧩 Training Pipeline Overview

1. **Data Loading & Augmentation:**
   - Dataset split into Train, Validation, and Test.
   - Augmentations applied only to Train set.

2. **Model Initialization:**
   - VGG-16 backbone with added layers.

3. **Hyperparameter Search:**
   - Grid Search over learning rates, batch sizes, and optimizers.
   - Cross-validation on each configuration.

4. **Training with Best Hyperparameters:**
   - Early stopping on validation loss.
   - LR scheduling applied.
   - Train & Validation metrics logged.S

5. **Evaluation:**
   - Final test accuracy.
   - Complete visualization report.
   - Optimizer comparison charts.

---

## 📊 Evaluation Results

The following plots will be generated and saved in the `results/` folder:

1. Training Loss Curve
2. Validation Loss Curve
3. Training Accuracy Curve
4. Validation Accuracy Curve
5. Learning Rate Curve
6. Hyperparameter Tuning Results Plot
7. Confusion Matrix
8. Per-Class Accuracy Bar Plot
9. Error Distribution per Class
10. ROC Curve per Class
11. Precision-Recall Curve per Class
12. F1 Score per Class
13. t-SNE Embedding Plot of Final Layer
14. Sample Misclassifications
15. Samples with Lowest Confidence
16. Samples with Highest Error Distance
17. Final Test Accuracy Comparison (AdamW vs SGD)
18. Validation Accuracy vs Epochs Comparison
19. Hyperparameter Grid Results Heatmap
20. Calibration Curve

---

## 👥 Contributors

- <a href="https://github.com/Lars314159">Lars Wunderlich</a> 
- <a href="https://github.com/ToniMahojoni">Toni Sand</a>
- <a href="https://github.com/hounaar">Parsa Besharat</a>

---

## 📝 License

This project is open-source and available under the MIT License.

