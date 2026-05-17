# 🍷 Wine Quality Prediction using Machine Learning and Streamlit

**Prepared for:** UMBC Data Science Master Degree  
**Instructor:** Dr. Chaojie (Jay) Wang  
**Author:** Akshay Kumar Nagabandi  

**Dataset:** Red Wine Quality Dataset (Cortez et al., 2009)

---

# 📌 Project Overview

Wine quality is traditionally evaluated by expert tasters based on sensory properties such as taste, aroma, and texture. However, this process is subjective, time-consuming, and may vary among individuals.

This project focuses on building and comparing multiple machine learning models to predict wine quality using physicochemical properties of red wine samples. The objective is to automate quality assessment using measurable chemical features rather than depending entirely on human evaluation.

The project also includes deployment using Streamlit, allowing users to interactively predict wine quality through a web application.

Project workflow includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Building
- Model Comparison
- Model Evaluation
- Streamlit Deployment

---

# 🎯 Project Objectives

- Perform exploratory data analysis (EDA)
- Understand feature relationships
- Identify important features affecting wine quality
- Train multiple classification models
- Compare model performance
- Select the best-performing model
- Deploy model using Streamlit

---

# 🔍 Research Questions

- Can physicochemical properties reliably predict wine quality using machine learning?
- Which model performs best for classification?
- Do ensemble models outperform simple classification models?
- Which features contribute most to prediction?

---

# 📊 Dataset Information

Dataset Source:

Kaggle Dataset:  
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Original Dataset:  
https://archive.ics.uci.edu/dataset/186/wine+quality

---

## Dataset Summary

| Metric | Value |
|----------|--------|
| Dataset Name | Red Wine Quality Dataset |
| Source | UCI / Kaggle |
| Total Samples | 1599 |
| Total Features | 11 |
| Target Variable | Quality |
| Quality Range | 0–10 |
| Input Type | Physicochemical Properties |
| Dataset Type | Structured Numerical Dataset |
| Wine Type | Portuguese Vinho Verde Red Wine |
| Data Nature | Real-world Dataset |
| Problem Type | Classification |
| Converted Target | Good Quality / Not Good Quality |
| Training Split | 70% |
| Testing Split | 30% |
| Missing Values | None |
| Class Distribution | Slightly Imbalanced |

---

# 📘 Attribute Information

## Input Features

1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

---

## Output Variable

Quality (0–10)

Converted target:

```text
goodquality = 1 if quality >= 7
goodquality = 0 if quality < 7
```

The original quality score was converted into binary classification:

- Good Quality → Quality ≥ 7
- Not Good Quality → Quality < 7

---

# ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Dataset loading and validation
- Missing value checking
- Binary target creation
- Feature-target separation
- Train-test split
- Data preparation for machine learning

Training and testing ratio:

- Training: 70%
- Testing: 30%

No missing values were found.

---

# 📊 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand data distributions and feature relationships.

EDA helps in:

- Understanding patterns
- Discovering relationships
- Identifying important variables
- Improving model understanding

---

## Quality Distribution Analysis

![Quality Distribution](quality_distribution.png)

### Observations

- Most wines belong to quality scores 5 and 6
- Very few wines belong to high-quality classes
- Dataset shows class imbalance
- Majority of samples are average quality

### Interpretation

This indicates that higher-quality wines occur less frequently in the dataset.

---

## Correlation Analysis

![Correlation Heatmap](correlation_heatmap.png)

### Observations

- Alcohol positively correlates with quality
- Volatile acidity negatively impacts quality
- Sulphates positively affect quality
- Some variables show weaker relationships

### Interpretation

Correlation analysis helps identify relationships between variables and target output.

---

# 📊 Feature Importance Analysis

Feature importance analysis determines which variables contribute most toward prediction.

![Feature Importance](feature_importance.png)

### Observations

- Alcohol is the most important feature
- Sulphates significantly contribute
- Volatile acidity strongly influences prediction
- Density and sulfur dioxide show moderate effects

### Interpretation

Understanding feature importance improves model interpretability.

---

# 🤖 Machine Learning Models Used

Models implemented:

### Random Forest Classifier

Advantages:

- Better accuracy
- Reduces overfitting
- Handles nonlinear relationships

---

### Decision Tree Classifier

Advantages:

- Easy interpretation
- Fast execution

---

### K-Nearest Neighbors (KNN)

Advantages:

- Simple implementation
- Effective classification approach

---

# 📈 Model Performance Comparison

Evaluation metrics used:

- Accuracy Score
- Confusion Matrix
- Classification Report

![Model Comparison](model_comparison.png)

### Observations

- Random Forest achieved highest accuracy
- Decision Tree performed moderately
- KNN showed comparatively lower performance

### Interpretation

Random Forest performed best because ensemble methods reduce overfitting and improve generalization.

---

# ⭐ Key Insights

- Alcohol positively impacts wine quality
- Volatile acidity negatively impacts quality
- Sulphates and citric acid improve quality
- Random Forest achieved best performance

---

# 🌐 Streamlit Deployment

The final machine learning model was deployed using Streamlit to create an interactive web application.

### Application Features

- Dataset overview
- Data visualization
- Model comparison
- Model performance results
- User input prediction form
- Real-time prediction
- Download prediction results

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Streamlit
- Jupyter Notebook
- VS Code

---

# 📂 Project Structure

```text
wine-quality-prediction/
│── app.py
│── Wine_Quality.ipynb
│── winequality-red.csv
│── requirements.txt
│── README.md
│── quality_distribution.png
│── correlation_heatmap.png
│── feature_importance.png
│── model_comparison.png
```

---

# 🚀 Future Work

- Hyperparameter tuning
- Handle class imbalance using SMOTE
- Feature selection techniques
- Model explainability using SHAP
- Deploy using Streamlit Cloud

---

# ⭐ Conclusion

This project demonstrates how machine learning techniques can effectively predict wine quality using physicochemical properties.

Different classification models were trained and compared, and among all models, Random Forest achieved the best performance.

The Streamlit application provides an interactive and user-friendly system for real-time wine quality prediction.

---

# Author

Akshay Kumar Nagabandi  
Master's Student  
University of Maryland, Baltimore County
