# Image Classification Pipeline

This project provides a complete pipeline for preprocessing, training, evaluation, and prediction using image data. Follow the steps below to configure and run the pipeline.

---

## 🔧 1. Data Augmentation & Normalization

To start augmenting and normalizing the raw dataset, run the following script:

```bash
python augmentation_init.py
```

You can modify the augmentation parameters in the `augmentation.py` file.

---

## ⚙️ 2. Configuration Setup

Before training or evaluation, make sure to configure the following paths:

### 2.1 `config.py`

Update the `BASE_PATH` variable to point to your working directory.  
For example, when using a cluster:

```python
BASE_PATH = "/home/username/..."
```

### 2.2 `load_dataset.py`

##### If you are using a cluster, you need to upload huge files to folder ```scratch ``` main folder and then your username subfolder.
Modify **line 16**:

```python
path = "/scratch/username"
```

### 2.3 `train.py`

Modify **line 37**:

```python
path = "/scratch/username"
```

### 2.4 `evaluate.py`

Modify **line 14**:

```python
path = "/scratch/username"
```

---

## 📁 3. Output Folders

Ensure the following folders exist and are empty before running the pipeline. They will be used to store results:

- `plots/` – Generated training and evaluation plots  
- `evaluation/` – Evaluation results  
- `best_params/` – Best hyperparameters for each optimizer  
- `augmented-db/` – Augmented image dataset  

---

## 🚀 4. Training & Hyperparameter Tuning

To perform training, hyperparameter tuning, and evaluation in one go, simply run:

```bash
python main.py
```

This will execute the full pipeline with cross-validation and optimizer comparison.

---

## 🧠 5. Predict New Images

At the end and after the model trained and gave you the plots, to predict the label of a new image using a trained model, run:

```bash
python predict_single_image.py
```

Make sure the model is already trained and saved.



