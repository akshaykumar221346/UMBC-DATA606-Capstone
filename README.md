# 🍷 Wine Quality Prediction using Machine Learning and Streamlit

Prepared for UMBC Data Science Master Degree

Dr. Chaojie (Jay) Wang

Author: Akshay Kumar Nagabandi

Dataset: UCI Wine Quality Dataset (Cortez et al., 2009)

---

## Project Overview

This project focuses on building and comparing multiple machine learning models to predict wine quality using physicochemical properties. The objective is to evaluate different classification algorithms and identify the most effective approach for wine quality prediction.

Wine quality is traditionally determined by human tasters, which is subjective and time-consuming. This project demonstrates how machine learning can automate quality assessment using measurable chemical features.

In addition, the project includes deployment of the final model using Streamlit, allowing users to interactively predict wine quality.

---

## Dataset Information

The dataset used in this project is the **Red Wine Quality Dataset (Cortez et al., 2009)**.

🔗 **Dataset Source (Kaggle):**
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

🔗 **Original Source (UCI Machine Learning Repository):**
https://archive.ics.uci.edu/dataset/186/wine+quality

### Dataset Description

The dataset contains physicochemical properties of Portuguese *Vinho Verde* red wine samples collected for quality assessment.

The goal of the dataset is to model wine quality based on measurable chemical features instead of subjective human evaluation.

### Dataset Characteristics

* 1599 wine samples
* 11 physicochemical input features
* 1 output variable (quality score between 0 and 10)
* Structured numerical dataset
* Real-world dataset
* Contains class imbalance

### Data Collection

* Collected in Portugal (Vinho Verde region)
* Quality scores assigned by wine experts
* Based on research paper:
  **Cortez et al., 2009 — Modeling wine preferences by data mining from physicochemical properties**

### Project-Specific Transformation

In this project, the problem is converted into a classification task:

```
goodquality = 1  if quality >= 7  
goodquality = 0  if quality < 7
```

---

## Attribute Information

### Input Features (Physicochemical Tests)

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

### Output Variable

* Quality (0–10)
* Converted Target: **Good Quality / Not Good Quality**

---

## Project Objectives

* Perform exploratory data analysis (EDA)
* Understand feature relationships
* Identify important features affecting wine quality
* Train multiple classification models
* Compare model performance
* Select best model
* Deploy model using Streamlit

---

## Main Research Questions

* Can physicochemical properties reliably predict wine quality using machine learning?
* Which model performs best for classification?
* Do ensemble models outperform simple models?
* Which features contribute most to prediction?

---

## Methodology

The project follows a structured workflow:

1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Model Comparison
8. Deployment using Streamlit

---

## Machine Learning Models Used

* Random Forest Classifier
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)

### Final Selected Model

* **Random Forest Classifier**

---

## Model Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## Key Insights

* Alcohol positively impacts wine quality
* Volatile acidity negatively impacts quality
* Sulphates and citric acid improve quality
* Random Forest performed best among all models

---

## Streamlit Deployment

The final model is deployed using Streamlit, transforming it into an interactive web application.

### App Features

* Dataset overview
* Data visualization (charts & heatmaps)
* Model comparison
* Final model performance
* User input prediction form
* Real-time prediction
* Download prediction result

---

## Project Structure

```
wine-quality-prediction/
│── app.py
│── Wine_Quality.ipynb
│── winequality-red.csv
│── requirements.txt
│── README.md
```

---

## How to Run the Project

### 1. Clone repository

```
git clone https://github.com/your-username/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run Streamlit app

```
streamlit run app.py
```

### 4. Open in browser

```
http://localhost:8501
```

---

## Sample Input (Good Quality Wine)

```
Fixed Acidity: 7.8
Volatile Acidity: 0.30
Citric Acid: 0.35
Residual Sugar: 2.0
Chlorides: 0.05
Free Sulfur Dioxide: 15
Total Sulfur Dioxide: 40
Density: 0.9968
pH: 3.30
Sulphates: 0.75
Alcohol: 11.5
```

Expected Output:
**Good Quality Wine**

---

## Future Work

* Hyperparameter tuning
* Handle class imbalance (SMOTE)
* Feature selection
* Model explainability (SHAP)
* Deploy online using Streamlit Cloud

---

## Libraries Used

* pandas
* numpy
* matplotlib
* scikit-learn
* streamlit

---

## Conclusion

This project demonstrates how machine learning can effectively predict wine quality using physicochemical properties. Among all models, **Random Forest achieved the best performance** and was selected for deployment.

---

## Author

Akshay Kumar Nagabandi
Master’s Student
UMBC
