# 🍷 Wine Quality Prediction using Machine Learning and Streamlit

## Project Title:
Wine Quality Prediction using Machine Learning and Streamlit

---

## Prepared for:
UMBC Data Science Master Degree Capstone  
Dr. Chaojie (Jay) Wang  

---

## Author Name:
Akshay Kumar Nagabandi

---

# 2. Background

## What is it about?

The current project aims to predict wine quality using machine learning techniques based on physicochemical properties of red wine samples. Traditional wine quality evaluation relies heavily on human experts and sensory testing methods, which can be subjective and time-consuming.

This project utilizes measurable chemical properties of wine samples to automate quality prediction using machine learning classification techniques.

Different machine learning algorithms are trained and evaluated to determine the best-performing model. The final model is deployed using Streamlit to create an interactive prediction application.

Unlike traditional approaches, this project provides a data-driven framework that can support efficient and consistent wine quality assessment.

---

## Why does it matter?

Wine quality prediction is important for wine manufacturers and quality control systems because it helps:

- Reduce dependency on manual evaluation
- Improve consistency in quality assessment
- Reduce time required for testing
- Support decision-making during production
- Improve customer satisfaction
- Enable scalable quality monitoring

---

## Research Questions

- Can physicochemical properties reliably predict wine quality using machine learning?
- Which machine learning model performs best for classification?
- Do ensemble models outperform traditional classification models?
- Which features contribute most to wine quality prediction?
- How do chemical properties influence wine quality?

---

# 3. Data

## Data Source

Variable:

**winequality-red.csv**  
(Red Wine Quality Dataset)

Dataset Source:

Kaggle:

https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Original Source:

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/186/wine+quality

---

## Dataset Description

The dataset consists of physicochemical properties of Portuguese Vinho Verde red wine samples collected for quality assessment.

The objective of the dataset is to model wine quality using measurable chemical properties rather than relying solely on subjective human evaluation.

---

## Dataset Characteristics

Format: CSV  

Total Rows: 1599  

Number of Columns: 12  

Input Features: 11  

Output Variable: Quality  

Dataset Type: Structured Numerical Dataset  

Wine Type: Portuguese Vinho Verde Red Wine  

Problem Type: Classification  

---

## Column Names and Description

| Column Name | Description |
|-------------|-------------|
| Fixed Acidity | Amount of fixed acids present in wine |
| Volatile Acidity | Amount of volatile acids present |
| Citric Acid | Citric acid concentration |
| Residual Sugar | Remaining sugar after fermentation |
| Chlorides | Salt concentration |
| Free Sulfur Dioxide | Free sulfur dioxide level |
| Total Sulfur Dioxide | Total sulfur dioxide level |
| Density | Density of wine |
| pH | Acidity/basicity measurement |
| Sulphates | Sulphate concentration |
| Alcohol | Alcohol percentage |
| Quality | Wine quality score |

---

# 4. Feature Engineering and Data Transformation

The original quality score is transformed into a binary classification target.

### Target Transformation

```text
goodquality = 1 if quality >= 7
goodquality = 0 if quality < 7
```

---

## Purpose of Transformation

- Simplify classification problem
- Improve model prediction capability
- Create practical quality categories

After transformation:

Each row corresponds to one wine sample.

The final dataset contains:

- Input physicochemical features
- Binary target variable

---

# 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis will be performed to understand relationships and patterns in the dataset.

The analysis includes:

- Missing value checking
- Statistical summary
- Quality distribution analysis
- Correlation analysis
- Feature importance analysis

Visualizations include:

- Quality distribution chart
- Correlation heatmap
- Feature importance chart
- Model comparison chart

Expected insights:

- Alcohol positively impacts quality
- Volatile acidity negatively impacts quality
- Sulphates contribute positively
- Some variables have weaker effects

---

# 6. Model Training

## Type of Learning

Supervised Learning — Classification

---

## Models to be Used

### Random Forest Classifier

Purpose:

- Handle nonlinear relationships
- Reduce overfitting

---

### Decision Tree Classifier

Purpose:

- Rule-based classification

---

### K-Nearest Neighbors (KNN)

Purpose:

- Instance-based classification

---

## Model Evaluation Metrics

Performance will be evaluated using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-score
- Classification Report

---

# 7. Trained Model Application

Application inputs:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

Application outputs:

- Predict wine quality
- Display probability score
- Download prediction result

---

## Example Prediction Interpretation

High Alcohol + High Sulphates + Low Volatile Acidity

→ Good Quality Wine

High Volatile Acidity + Low Alcohol

→ Not Good Quality Wine

---

This demonstrates practical application of machine learning for automated quality prediction.

---

# 8. Conclusion

Wine quality prediction is an important task in the food and beverage industry because it helps improve quality assessment and decision-making processes.

This project demonstrates how machine learning techniques can effectively predict wine quality using measurable chemical properties rather than relying entirely on human judgment.

Among the evaluated models, Random Forest is expected to provide the best performance and will be used for deployment through Streamlit to support real-time interactive prediction.
