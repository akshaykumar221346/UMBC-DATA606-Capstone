import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

st.markdown("""
<style>


            
/* Space between tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

/* Normal tab style */
.stTabs [data-baseweb="tab"] {
    height: 50px;
    border-radius: 12px;
    padding: 0 20px;
    background-color: #f5f5f5;
    color: #333333;
    font-weight: 600;
    border: 1px solid #dddddd;
    transition: all 0.3s ease;
}

/* Hover effect */
.stTabs [data-baseweb="tab"]:hover {
    background-color: #eadfe3;
    color: #8B1E3F;
    border: 1px solid #c9a7b3;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8B1E3F, #D4AF37) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(139, 30, 63, 0.25);
}

/* Make active tab text bold */
.stTabs [aria-selected="true"] p {
    font-weight: 700;
}

/* Optional: smoother tab text */
.stTabs [data-baseweb="tab"] p {
    font-size: 15px;
}

            
/* Predict Button Style */
.stButton > button {
    background: linear-gradient(135deg, #8B1E3F, #D4AF37);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 10px 18px;
    width: 100%;
    transition: all 0.3s ease;
}

/* Hover Effect */
.stButton > button:hover {
    background: linear-gradient(135deg, #6A152F, #b8962e);
    transform: scale(1.03);
    box-shadow: 0 6px 14px rgba(139, 30, 63, 0.3);
}

/* Click Effect */
.stButton > button:active {
    transform: scale(0.97);
}
    
</style>
""", unsafe_allow_html=True)

DATA_PATH = "winequality-red.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["goodquality"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)
    return df


@st.cache_resource
def train_models(df):
    X = df.drop(["quality", "goodquality"], axis=1)
    y = df["goodquality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=7
    )

    models = {
        "Random Forest": RandomForestClassifier(random_state=1),
        "Decision Tree": DecisionTreeClassifier(random_state=1),
        "KNN": KNeighborsClassifier()
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        trained_models[name] = model
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4)
        })

    final_model = trained_models["Random Forest"]
    final_pred = final_model.predict(X_test)
    final_acc = accuracy_score(y_test, final_pred)
    final_cm = confusion_matrix(y_test, final_pred)
    final_report = classification_report(y_test, final_pred, output_dict=True)

    return {
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "model_results": pd.DataFrame(results).sort_values(by="Accuracy", ascending=False),
        "final_model": final_model,
        "final_acc": final_acc,
        "final_cm": final_cm,
        "final_report": pd.DataFrame(final_report).transpose()
    }


try:
    wine = load_data()
    artifacts = train_models(wine)
except FileNotFoundError:
    st.error("Please keep 'winequality-red.csv' in the same folder as app.py")
    st.stop()

X = artifacts["X"]
y = artifacts["y"]
model_results = artifacts["model_results"]
model = artifacts["final_model"]
acc = artifacts["final_acc"]
cm = artifacts["final_cm"]
report_df = artifacts["final_report"]

st.title("🍷 Wine Quality Prediction using Machine Learning")
st.markdown(
    """
This project predicts whether a red wine is **Good Quality** or **Not Good Quality**
using machine learning.

**Target rule used in this project:**  
- `quality >= 7` → Good Quality  
- `quality < 7` → Not Good Quality
"""
)

with st.sidebar:
    st.header("Project Summary")
    st.write("**Dataset:** Wine Quality Dataset")
    st.write("**Type:** Classification")
    st.write("**Final Model:** Random Forest")
    st.write(f"**Final Accuracy:** {acc:.4f}")
    st.write("**Goal:** Predict wine quality from chemical properties")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Project Overview",
    "Dataset Analysis",
    "Model Comparison",
    "Final Model Results",
    "Prediction"
])

with tab1:
    st.subheader("Project Description")
    st.write(
        """
This application is based on the Wine Quality Prediction project.
The dataset contains physicochemical properties of red wine such as acidity,
chlorides, sulphates, alcohol, pH, and density.

The main objective of this project is to predict whether a wine sample belongs
to the good quality category or not using machine learning models.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", wine.shape[0])
    c2.metric("Columns", wine.shape[1])
    c3.metric("Good Quality Wines", int(wine["goodquality"].sum()))
    c4.metric("Not Good Quality Wines", int((wine["goodquality"] == 0).sum()))

    st.write("### Dataset Preview")
    st.dataframe(wine.head(10), use_container_width=True)

    st.write("### Missing Values")
    st.dataframe(wine.isnull().sum().to_frame("Missing Values"), use_container_width=True)

    st.write("### Statistical Summary")
    st.dataframe(wine.describe(), use_container_width=True)

with tab2:
    st.subheader("Dataset Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Quality Distribution")
        fig1, ax1 = plt.subplots()
        wine["quality"].value_counts().sort_index().plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Quality Score")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of Wine Quality")
        st.pyplot(fig1)

    with col2:
        st.write("### Good vs Not Good Quality")
        fig2, ax2 = plt.subplots()
        wine["goodquality"].value_counts().sort_index().plot(kind="bar", ax=ax2)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["Not Good", "Good"])
        ax2.set_ylabel("Count")
        ax2.set_title("Binary Quality Distribution")
        st.pyplot(fig2)

    st.write("### Correlation Heatmap")
    corr = wine.corr(numeric_only=True)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    img = ax3.imshow(corr, aspect="auto")
    ax3.set_xticks(range(len(corr.columns)))
    ax3.set_xticklabels(corr.columns, rotation=90)
    ax3.set_yticks(range(len(corr.columns)))
    ax3.set_yticklabels(corr.columns)
    fig3.colorbar(img)
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)
    st.write("### Feature Importance")
    et_model = ExtraTreesClassifier(random_state=42)
    et_model.fit(X, y)
    importance = pd.Series(
        et_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    importance.plot(
        kind="barh",
        ax=ax4,
        color=["#8B1E3F", "#A52A2A", "#C04060", "#D36C8A", "#E6A4B4",
            "#7A9E9F", "#5F8D8D", "#4F6F52", "#739072", "#AA8B56", "#D4AF37"]
    )

    ax4.set_xlabel("Importance Score")
    ax4.set_ylabel("Input Features")
    ax4.set_title("Feature Importance")
    ax4.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    st.pyplot(fig4)




with tab3:
    st.subheader("Model Comparison")
    st.write(
        "The models below were trained on the same train-test split. "
        "Random Forest is used as the final model because it performed best."
    )
    st.dataframe(model_results, use_container_width=True)

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    colors = ["#8B1E3F", "#D4AF37", "#739072"]

    ax5.bar(
        model_results["Model"],
        model_results["Accuracy"],
        color=colors
    )

    ax5.set_ylabel("Accuracy")
    ax5.set_title("Accuracy Comparison of Models")
    ax5.tick_params(axis="x", labelrotation=0)
    plt.tight_layout()
    st.pyplot(fig5)

with tab4:
    st.subheader("Final Model Results")
    st.metric("Random Forest Accuracy", f"{acc:.4f}")

    st.write("### Confusion Matrix")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Not Good", "Actual: Good"],
        columns=["Predicted: Not Good", "Predicted: Good"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.write("### Classification Report")
    st.dataframe(report_df, use_container_width=True)

    st.write("### Conclusion")
    st.success(
        "Random Forest performed best for this dataset and was selected as the final model "
        "for wine quality prediction."
    )

with tab5:
    st.subheader("Predict Wine Quality")
    st.write("Enter the wine features below to get the prediction:")

    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", value=7.8, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity", value=0.30, step=0.01)
        citric_acid = st.number_input("Citric Acid", value=0.35, step=0.01)
        residual_sugar = st.number_input("Residual Sugar", value=2.0, step=0.1)
        chlorides = st.number_input("Chlorides", value=0.076, step=0.050, format="%.3f")
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15.0, step=1.0)

    with col2:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=40.0, step=1.0)
        density = st.number_input("Density", value=0.9968, step=0.0001, format="%.4f")
        ph = st.number_input("pH", value=3.30, step=0.01)
        sulphates = st.number_input("Sulphates", value=0.75, step=0.01)
        alcohol = st.number_input("Alcohol", value=11.5, step=0.1)

    input_df = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [ph],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })

    if st.button("Predict Quality"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        st.write("### Entered Values")
        st.dataframe(input_df, use_container_width=True)

        if pred == 1:
            result_text = "Good Quality Wine"
            st.success(f"Prediction: {result_text}")
        else:
            result_text = "Not Good Quality Wine"
            st.error(f"Prediction: {result_text}")

        st.write(f"**Probability of Good Quality:** {prob[1]:.4f}")
        st.write(f"**Probability of Not Good Quality:** {prob[0]:.4f}")

        download_df = input_df.copy()
        download_df["Prediction"] = result_text
        download_df["Probability_Good"] = round(float(prob[1]), 4)
        download_df["Probability_Not_Good"] = round(float(prob[0]), 4)

        csv = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Result",
            data=csv,
            file_name="wine_prediction_result.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Capstone Project- Wine Quality Prediction using Streamlit")