# 🌍 Air Quality Prediction Model

A Machine Learning project for predicting **Air Quality** using the **AirQualityUCI dataset**. The project implements a complete ML workflow including data ingestion, data preprocessing, feature transformation, model training, prediction, evaluation, and cross-validation.

---

## 📌 Project Overview

Air pollution is an important environmental issue that can be monitored using air-quality sensor measurements.

The objective of this project is to build a **Regression Machine Learning model** that predicts the `AH` target variable from the available air-quality features.

The project uses:

* Python
* Pandas
* Scikit-learn
* Random Forest Regression
* FLAML
* Excel dataset
* Machine Learning pipelines
* Cross-validation

---

## 🎯 Objective

The main objective is to develop a regression model that can accurately predict the target variable `AH` based on air-quality measurements.

The workflow includes:

1. Loading the dataset
2. Removing duplicate records
3. Separating features and target
4. Removing unwanted date/time columns
5. Splitting data into training and testing sets
6. Handling missing values
7. Scaling numerical features
8. Encoding categorical features
9. Training a Random Forest Regression model
10. Predicting values on the test dataset
11. Evaluating the model using R² score
12. Performing 5-fold cross-validation

---

## 📂 Project Structure

```text
AQI_PredicationModel/
│
├── data/
│   └── AirQualityUCI.xlsx
│
├── src/
│   └── aqi_predicationmodel/
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       └── model_build.py
│
├── main.py
├── README.md
└── pyproject.toml
```

---

## 📊 Dataset

The project uses the **AirQualityUCI.xlsx** dataset.

The data contains air-quality measurements collected from sensors. The model uses the available feature columns to predict:

```text
Target Variable: AH
```

The columns `Date` and `Time` are removed during preprocessing because they are not directly used as model input features.

---

## 🔄 Machine Learning Workflow

```text
             AirQualityUCI.xlsx
                     │
                     ▼
              Data Ingestion
                     │
                     ▼
             Remove Duplicates
                     │
                     ▼
          Feature / Target Split
                     │
                     ▼
              Train-Test Split
                     │
                     ▼
        ┌────────────┴────────────┐
        ▼                         ▼
 Numerical Features       Categorical Features
        │                         │
 Median Imputation       Most-Frequent Imputation
        │                         │
 RobustScaler             OneHotEncoder
        └────────────┬────────────┘
                     ▼
             ColumnTransformer
                     │
                     ▼
          Random Forest Regressor
                     │
                     ▼
                 Prediction
                     │
                     ▼
                R² Score
                     │
                     ▼
             5-Fold Cross Validation
```

---

# 🛠️ Technologies & Libraries

## Python

Python is used as the primary programming language for implementing the complete machine-learning pipeline.

## Pandas

Used for loading and handling the Excel dataset.

```python
import pandas as pd
```

The project loads the dataset using:

```python
df = pd.read_excel(...)
```

### Why Pandas?

Pandas makes it easy to:

* Load datasets
* Inspect data
* Remove duplicates
* Select columns
* Handle structured/tabular data

---

## Scikit-learn

Scikit-learn is used for preprocessing, splitting the dataset, model training, and evaluation.

The project uses:

* `Pipeline`
* `train_test_split`
* `RobustScaler`
* `OneHotEncoder`
* `SimpleImputer`
* `ColumnTransformer`
* `RandomForestRegressor`
* `KFold`
* `cross_val_score`
* `r2_score`

---

# 🧹 Data Preprocessing

## 1. Remove Duplicate Records

```python
df = df.drop_duplicates()
```

Duplicate records can cause the model to see the same information multiple times and can negatively affect model evaluation.

---

## 2. Separate Features and Target

The target variable is:

```text
AH
```

The following columns are removed from the features:

```text
Date
Time
AH
```

```python
X = df.drop(columns=['AH','Date','Time'])
y = df['AH']
```

Here:

* `X` → input features
* `y` → target variable

---

## 3. Train-Test Split

The dataset is divided into:

* **70% Training data**
* **30% Testing data**

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1
)
```

### Why Train-Test Split?

The training data is used to learn patterns, while the test data is used to evaluate how well the trained model performs on unseen data.

---

# 🔢 Numerical Data Processing

Numerical features use the following pipeline:

```python
numerical_pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='median')),
    ('Scaler', RobustScaler())
])
```

### Median Imputation

Missing numerical values are replaced with the median.

### Why Median?

Median is less affected by extreme values/outliers than the mean.

### RobustScaler

RobustScaler scales numerical data using statistics that are more resistant to outliers.

This is useful for air-quality data because sensor measurements can contain unusually high or low values.

---

# 🔤 Categorical Data Processing

Categorical features use:

```python
categorical_pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='most_frequent')),
    ('Encoder', OneHotEncoder(
        drop='first',
        handle_unknown='ignore',
        sparse_output=False
    ))
])
```

### Most-Frequent Imputation

Missing categorical values are replaced with the most frequently occurring category.

### One-Hot Encoding

Categorical values are converted into numerical representation so that the Machine Learning model can process them.

### `drop='first'`

The first category is removed to reduce redundant encoded columns.

### `handle_unknown='ignore'`

If an unseen category appears in the test data, the encoder will not throw an error.

---

# 🔀 ColumnTransformer

The numerical and categorical pipelines are combined using:

```python
ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_data),
        ('cat', categorical_pipeline, categorical_data)
    ]
)
```

This allows different preprocessing techniques to be applied automatically to different types of columns.

---

# 🌲 Machine Learning Model

## Random Forest Regressor

The main model used in this project is:

```python
RandomForestRegressor(n_estimators=500)
```

### Why Random Forest?

Random Forest is a strong choice for regression because:

* It can capture nonlinear relationships.
* It combines multiple decision trees.
* It generally performs well on tabular data.
* It can model complex relationships between air-quality features.
* It is less dependent on feature scaling compared with many distance-based algorithms.

---

## 🌳 Number of Trees

The model uses:

```text
n_estimators = 500
```

This means the Random Forest contains **500 decision trees**.

Using many trees can make predictions more stable and reduce the variance of the model, although increasing the number of trees also increases computational cost.

---

# 📈 Model Evaluation

The model uses **R² (R-squared)** to evaluate regression performance.

```python
r2score = r2_score(y_pred, y_test)
```

### R² Score

R² measures how well the model explains the variation in the target variable.

A simplified interpretation:

```text
R² = 1.0   → Excellent fit
R² = 0.0   → Model does not explain the variance
R² < 0     → Model performs poorly
```

The project achieved an approximately **99% R² score** during model evaluation.

> Note: In the implementation, the arguments passed to `r2_score` are `y_pred, y_test`. For standard scikit-learn usage, the conventional order is `r2_score(y_test, y_pred)`.

---

# 🔁 Cross-Validation

The project also performs **5-fold cross-validation**.

```python
kf = KFold(
    n_splits=5,
    shuffle=True
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kf,
    scoring="r2"
)
```

### Why Cross-Validation?

A single train-test split may not provide a complete picture of model performance.

5-fold cross-validation:

1. Divides the training data into 5 parts.
2. Trains the model on 4 parts.
3. Validates it on the remaining part.
4. Repeats this process 5 times.
5. Calculates the average R² score.

The project prints both individual cross-validation scores and their mean.

---

# 🤖 FLAML

The project also imports:

```python
from flaml.automl import AutoML
```

FLAML is an AutoML framework that can automatically search for suitable machine-learning models and hyperparameters.

The current `model_build.py` imports `AutoML`, but the actual model currently implemented is `RandomForestRegressor`.

---

# ▶️ How to Run the Project

## 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd AQI_PredicationModel
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install pandas scikit-learn openpyxl flaml
```

## 5. Run the project

```bash
python main.py
```

The main file loads the dataset, performs preprocessing, trains the model, generates predictions, and prints the R² and cross-validation results.

---

# 📊 Expected Output

The program displays information such as:

```text
Dataset shape
Training data shape
Testing data shape

the r2score is ...
Cross-validation scores: [...]
Mean cross-validation R2 score: ...
```

---

# 💡 Key Learning Outcomes

Through this project, I learned how to:

* Build an end-to-end Machine Learning pipeline
* Work with real-world air-quality data
* Perform data preprocessing
* Handle missing values
* Encode categorical variables
* Scale numerical features
* Split data into training and testing datasets
* Build a Random Forest regression model
* Evaluate regression performance using R²
* Perform K-Fold cross-validation
* Organize an ML project into separate modules

---

# 🚀 Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Comparing Random Forest with other regression algorithms
* Implementing FLAML AutoML
* Adding MAE and RMSE metrics
* Saving the trained model using Pickle/Joblib
* Creating a Streamlit or Gradio interface
* Adding MLflow experiment tracking
* Deploying the model as an API
* Creating visualizations for actual vs predicted values

---

# 👨‍💻 Author

**Mayur Wabale**

Machine Learning | Data Analytics | Python | SQL | Power BI

---

## ⭐ Project Summary

**Air Quality Prediction Model** is an end-to-end regression project that uses air-quality sensor data and a **Random Forest Regressor** to predict the `AH` target variable. The project demonstrates practical implementation of data preprocessing, feature transformation, machine learning, evaluation, and cross-validation.
# AQI_PredicationModel
