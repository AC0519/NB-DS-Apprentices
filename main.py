import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    median_ages = df.groupby('Pclass')['Age'].median()
    df['Age'] = df.apply(lambda row: median_ages[row['Pclass']] if pd.isna(row['Age']) else row['Age'], axis=1)
    return df

df = load_data()

st.title("Predicting Who Died on the Titanic: A Naive Bayes Approach")

st.markdown('Data Source can be found <a href="https://www.kaggle.com/datasets/yasserh/titanic-dataset">here</a>', 
            unsafe_allow_html=True)

with st.expander("See full data table"):
    st.write(df)

with st.form("feature selection"):
    col1, col2, col3 = st.columns(3)

    with col1:
        features = st.multiselect(
            "Select features to include in the model:",
            options=[col for col in df.columns if col != "Survived"]
        )
                
    with col2:
        if features:
            st.write("Histograms of Selected Features:")
            for feature in features:
                fig, ax = plt.subplots()
                df[feature].hist(ax=ax)
                ax.set_title(f'Histogram of {feature}')
                st.pyplot(fig)
    
    with col3:
        model_type = st.selectbox(
            "Select Naive Bayes Model:",
            options=["GaussianNB", "MultinomialNB", "BernoulliNB"]
        )
    
    submitted = st.form_submit_button("Submit")

if submitted:
    @st.cache_resource
    def initialize_model(model_type):
        if model_type == "GaussianNB":
            return GaussianNB()
        elif model_type == "MultinomialNB":
            return MultinomialNB()
        elif model_type == "BernoulliNB":
            return BernoulliNB()

    model = initialize_model(model_type)
    st.write(f"Initialized model: {model}")

    X = df[features]
    y = df["Survived"]

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate F1 score, precision, and recall
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    tab1, tab2 = st.tabs(["Model Metrics", "Confusion Matrix"])

    with tab1:
        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")

    with tab2:
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
