# Passive Sensor-Based Wellbeing Prediction

A machine learning research project for **predicting next-day wellbeing indicators from passive smartphone sensing data**.

The project is based on the **StudentLife dataset** and investigates whether behavioral signals collected passively from a smartphone—such as calls, SMS activity, Wi-Fi interactions, and application usage—can be used to predict three wellbeing dimensions:

- **Stress**
- **Mood**
- **Health / Sleep**

The pipeline is designed to reproduce the core experimental setup of a research paper while replacing the original deep-learning approach with several traditional machine-learning models for comparison.

---

## 📌 Project Overview

Passive sensing allows behavioral information to be collected without requiring users to manually record every activity. This project transforms raw StudentLife sensing data into daily behavioral features and uses a **7-day observation window** to predict wellbeing on the following day.

### Prediction tasks

| Target | Description |
|---|---|
| **Stress** | Stress level derived from EMA stress responses |
| **Mood** | Mood score derived from positive and negative behavioral EMA responses |
| **Health** | Sleep-related health score derived from sleep duration and sleep quality |

All target scores are normalized to approximately **0–100**, where higher values represent greater levels of the corresponding positive/negative construct according to the target definition.

---

## 🔬 Methodology

The complete pipeline consists of four stages:

```text
StudentLife Dataset
        │
        ▼
┌─────────────────────┐
│ 1. Data Preparation │
└─────────────────────┘
        │
        ▼
Daily passive-sensing features
        │
        ▼
┌────────────────────────┐
│ 2. Feature Engineering │
└────────────────────────┘
        │
        ▼
7-day aggregated features
        │
        ▼
┌───────────────────┐
│ 3. Model Training │
└───────────────────┘
        │
        ▼
Stress / Mood / Health
        │
        ▼
┌─────────────────┐
│ 4. Evaluation   │
└─────────────────┘
```

### 1. Data preparation

`01_prepare_data.py` loads the StudentLife data and converts raw records into a daily participant-level dataset.

Passive sensing sources include:

- 📞 Call logs
- 💬 SMS activity
- 📶 Wi-Fi sensing
- 📱 Application usage

EMA responses are used to construct the prediction targets.

The script creates:

```text
data/processed/raw_daily.csv
data/processed/supervised.csv
```

The supervised dataset uses a **7-day historical window** and predicts the target for the following day.

### 2. Feature engineering

`02_feature_engineering.py` performs:

- Participant-independent **80/20 train/test split**
- Missing-value imputation using `IterativeImputer`
- Feature normalization using `StandardScaler`
- Reproducible splitting with random seed `42`

The split is performed by participant rather than by individual samples, helping prevent observations from the same participant appearing in both training and testing sets.

Generated artifacts include:

```text
data/processed/X_train.npy
data/processed/X_test.npy
data/processed/y_train.pkl
data/processed/y_test.pkl
data/processed/imputer.pkl
data/processed/scaler.pkl
data/processed/feature_cols.pkl
```

### 3. Model training

`03_train_models.py` trains and compares multiple regression approaches:

- **Ridge Regression**
- **Support Vector Regression (SVR) with RBF kernel**
- **Random Forest Regression**
- **Gradient Boosting Regression**
- **XGBoost Regression**
- **Multi-task Ridge Regression**

Hyperparameters for the individual models are selected using **5-fold cross-validation** on the training participants.

The primary evaluation metric is:

> **Mean Absolute Error (MAE)**

Lower MAE indicates better prediction performance.

Trained models and prediction results are stored in the `models/` directory.

### 4. Evaluation and analysis

`04_evaluate.py` generates several analyses:

- Model comparison against paper-reported LSTM baselines
- Transfer-learning simulation
- Gradient Boosting feature-importance analysis
- Predicted-vs-actual plots
- Per-participant stability versus MAE analysis

Outputs are saved under:

```text
results/
```

---

## 📊 Features

The project extracts daily behavioral statistics from passive sensing streams.

### Call features

- Number of calls
- Total call duration
- Mean call duration

### SMS features

- Number of SMS messages

### Wi-Fi features

- Number of unique access points
- Mean Wi-Fi signal level
- Standard deviation of Wi-Fi signal level
- Number of Wi-Fi scans

### Application usage features

- Number of application events
- Number of unique applications

For each sensing feature, the 7-day window is summarized using:

- **Mean**
- **Standard deviation**
- **Most recent value**

The current-day wellbeing score is also included as a lagged feature when available.

---

## 📁 Repository Structure

```text
Passive-Sensor-Based-Wellbeing-Prediction/
│
├── models/
│   └── Trained model artifacts
│
├── 01_prepare_data.py
│   └── Load StudentLife data and build supervised dataset
│
├── 02_feature_engineering.py
│   └── Participant split, imputation and normalization
│
├── 03_train_models.py
│   └── Train and tune machine-learning models
│
├── 04_evaluate.py
│   └── Evaluation, comparison and visualization
│
├── evaluate.py
│   └── Additional evaluation utilities
│
├── data/
│   └── processed/
│       └── Generated datasets and preprocessing artifacts
│
├── results/
│   └── Generated evaluation results and plots
│
└── README.md
```

> `data/`, `models/`, and `results/` may contain generated artifacts depending on how the project is configured locally.

---

## 🧰 Requirements

The project uses Python and the following major libraries:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- XGBoost

Install the dependencies with:

```bash
pip install numpy pandas scikit-learn joblib matplotlib seaborn xgboost
```

---

## 📥 Dataset

This project uses the **StudentLife dataset** from Dartmouth College.

The dataset contains smartphone-based sensing and ecological momentary assessment (EMA) data collected from participants during the StudentLife study.

You must obtain the dataset separately and place it locally before running the pipeline.

The expected dataset structure includes directories such as:

```text
dataset/
├── EMA/
├── call_log/
├── sms/
├── sensing/
└── app_usage/
```

The exact path is configured in:

```text
01_prepare_data.py
```

Update:

```python
DATASET_DIR = Path("path/to/StudentLife/dataset")
```

to point to your local StudentLife dataset.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/vipulbhati660/Passive-Sensor-Based-Wellbeing-Prediction.git
cd Passive-Sensor-Based-Wellbeing-Prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn joblib matplotlib seaborn xgboost
```

### 3. Configure the dataset path

Open:

```text
01_prepare_data.py
```

and set `DATASET_DIR` to the location of your StudentLife dataset.

### 4. Prepare the data

```bash
python 01_prepare_data.py
```

This creates the daily and supervised datasets in:

```text
data/processed/
```

### 5. Perform feature engineering

```bash
python 02_feature_engineering.py
```

This creates the train/test arrays and preprocessing artifacts.

### 6. Train the models

```bash
python 03_train_models.py
```

The script performs model selection using 5-fold cross-validation and evaluates the selected models on held-out participants.

### 7. Generate evaluation results

```bash
python 04_evaluate.py
```

Plots and analysis outputs are written to:

```text
results/
```

---

## 📈 Evaluation Metric

The primary metric is **Mean Absolute Error (MAE)**:

```text
MAE = (1/n) × Σ |yᵢ − ŷᵢ|
```

where:

- `yᵢ` is the actual wellbeing score
- `ŷᵢ` is the predicted wellbeing score
- `n` is the number of evaluated samples

A lower MAE indicates that predictions are closer to the observed wellbeing scores.

---

## 🔁 Reproducibility

The experiments use a fixed random seed:

```python
SEED = 42
```

The train/test split is performed at the **participant level**, rather than randomly splitting individual daily observations.

This is important for evaluating whether the models generalize to participants that were not used during training.

---

## 🧪 Research Comparison

The project also provides a comparison with the LSTM results reported in the reference paper.

The training script records the paper's reported values for:

| Model | Stress MAE | Mood MAE | Health MAE |
|---|---:|---:|---:|
| Paper — Deep LSTM | 16.80 | 15.70 | 15.60 |
| Paper — Transfer Learning LSTM | 14.40 | 13.50 | 13.20 |

These values are used as reference baselines for the experiments in this repository.

> **Important:** The paper-reported results and results produced by this repository should not be interpreted as directly equivalent models. The repository uses traditional machine-learning models and engineered statistical features, while the reference approaches use LSTM-based architectures.

---

## 🧠 Why Passive Sensing?

Traditional wellbeing assessments often rely heavily on questionnaires and self-reported information. Passive sensing provides another source of behavioral information by continuously observing patterns such as communication, mobility-related Wi-Fi activity, and smartphone usage.

This project explores whether these behavioral signals contain enough information to estimate short-term changes in wellbeing.

The broader research question is:

> **Can passive smartphone sensing data be used to predict an individual's next-day wellbeing?**

---

## ⚠️ Limitations

This project is intended as a **research and machine-learning experimentation project**, not as a medical diagnostic system.

Important limitations include:

- StudentLife is a relatively small and specific study population.
- EMA responses are used as proxy labels for wellbeing.
- Passive sensing data can contain missing or noisy observations.
- Results may not generalize to other populations or environments.
- Traditional ML models operate on engineered statistical features rather than the original raw temporal sequences.
- Prediction performance should not be interpreted as clinical diagnosis or medical advice.

---

## 🔮 Future Work

Potential extensions include:

- Implementing the original LSTM architecture
- Reproducing the paper's transfer-learning approach
- Comparing traditional ML against deep temporal models
- Adding additional StudentLife sensing modalities
- Using personalized models for individual participants
- Exploring temporal models such as GRU, Transformer and Temporal Fusion Transformer
- Applying explainable AI techniques such as SHAP
- Performing more robust participant-level cross-validation
- Evaluating model calibration and uncertainty
- Investigating domain adaptation across different datasets

---

## 📚 Research Context

The repository is part of an investigation into **passive smartphone sensing and wellbeing prediction**, with a focus on reproducing and extending research results using alternative machine-learning approaches.

The implementation emphasizes:

1. Reproducible preprocessing
2. Participant-independent evaluation
3. Multiple baseline models
4. Quantitative comparison using MAE
5. Feature-level analysis
6. Comparison with published LSTM baselines

---

## 👤 Author

**Vipul Bhati**

GitHub:  
https://github.com/vipulbhati660

Repository:  
https://github.com/vipulbhati660/Passive-Sensor-Based-Wellbeing-Prediction

---

## 📄 License

Please refer to the repository for the applicable license.

---

## ⭐ Acknowledgements

This project uses the **StudentLife dataset** and builds upon research in passive smartphone sensing, behavioral modeling, and wellbeing prediction.

If you find this project useful for research or learning, consider ⭐ starring the repository.
