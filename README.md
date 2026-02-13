# 🍷 Comparative Analysis of Ensemble Machine Learning Models for Wine Quality Classification

## Project Overview

This project focuses on building and comparing multiple machine learning models to predict wine quality using physicochemical properties. The objective is to evaluate different classification algorithms and identify the most effective approach for wine quality prediction.

Wine quality is traditionally determined by human tasters, which is subjective and time-consuming. This project demonstrates how machine learning can automate quality assessment using measurable chemical features.

---

## Dataset Information

The dataset consists of red and white variants of Portuguese *Vinho Verde* wine. It is publicly available from the UCI Machine Learning Repository and Kaggle.

🔗 **Dataset Source:**  
https://www.kaggle.com/rajyellow46/wine-quality

### Dataset Characteristics

- ~6,400+ wine samples  
- 11 physicochemical input features  
- 1 output variable (quality score between 0 and 10)  
- Structured numerical dataset  
- Contains class imbalance  
- Real-world industry dataset  

The red and white wine datasets were combined. A few values were randomly removed to simulate realistic data conditions.

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

12. Quality (Score between 0 and 10)

---

## Project Objectives

- Perform exploratory data analysis (EDA)
- Identify key features affecting wine quality
- Train and compare multiple classification models
- Evaluate performance using standard metrics
- Optimize models for better accuracy

---

## Methodology

The project follows a structured machine learning workflow:

1. Data Loading & Cleaning  
2. Exploratory Data Analysis  
3. Feature-Target Separation  
4. Train-Test Split  
5. Model Training  
6. Model Evaluation  
7. Performance Comparison  

---

## Machine Learning Algorithms Used

The following classification models were implemented and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Extra Trees  
- XGBoost  
- LightGBM  

These models were chosen to compare:

- Linear models  
- Tree-based models  
- Ensemble methods  
- Boosting algorithms  

---

## Results

- Alcohol shows strong positive correlation with wine quality  
- Volatile acidity shows negative correlation  
- Ensemble and boosting models outperform basic linear models  
- Best model achieved approximately **89% accuracy**

---

## Challenges

- Class imbalance (more average-quality wines than extreme ones)
- Potential outliers
- Ordered target variable
- Possible irrelevant features

---

## Future Work

- Outlier detection and removal  
- Feature selection techniques  
- Hyperparameter tuning (GridSearch / RandomizedSearch)  
- Random UnderSampling or SMOTE for imbalance handling  
- Deployment using Streamlit  
- Model explainability using SHAP  

---

## Libraries Used

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- XGBoost  
- LightGBM  

---

## Project Structure

```
├── data/
│   └── winequality.csv
├── notebooks/
│   └── Wine_Quality_Prediction_Analysis.ipynb
├── models/
│   └── best_model.pkl
└── README.md
```

---

## Conclusion

This project demonstrates an end-to-end machine learning pipeline for wine quality classification. It highlights the effectiveness of ensemble learning techniques and provides insights into the chemical properties that influence wine quality.

---
