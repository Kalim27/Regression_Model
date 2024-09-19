 
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Title of the Streamlit App
st.title("Basic Linear Regression App")

# Instructions
st.write("""
### Upload your dataset (CSV format) to build a linear regression model.
Ensure your dataset contains numerical columns for features and target values.
""")

# File uploader for CSV dataset
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.write(data.head())

    label_encoder = LabelEncoder()
    data['Brand Enc'] = label_encoder.fit_transform(data['Brand'])
    data['Model Enc'] = label_encoder.fit_transform(data['Model'])
    data['Fuel Enc'] = label_encoder.fit_transform(data['Fuel'])
    data['Seller_Type Enc'] = label_encoder.fit_transform(data['Seller_Type'])
    data['Transmission Enc'] = label_encoder.fit_transform(data['Transmission'])
    data['Owner Enc'] = label_encoder.fit_transform(data['Owner'])

    st.write(data.head())

    # Select target and feature columns
    st.write("### Select the target variable (y) and features (X):")
    all_columns = data.columns.tolist()
    target_column = st.selectbox("Select target variable (y):", all_columns)
    feature_columns = st.multiselect("Select feature variables (X):", all_columns)

    if st.button("Train Model"):
        if target_column and feature_columns:
            X = data[feature_columns]
            y = data[target_column]

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create and train the linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Show model coefficients
            st.write("### Model Coefficients:")
            for i, col in enumerate(feature_columns):
                st.write(f"{col}: {model.coef_[i]:.4f}")

            st.write(f"Intercept: {model.intercept_:.4f}")

            # Model performance metrics
            st.write("### Model Performance:")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
            st.write(f"R-squared: {r2_score(y_test, y_pred):.4f}")

            # Plot actual vs predicted values
            st.write("### Actual vs Predicted Plot:")
            plt.scatter(y_test, y_pred)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Actual vs Predicted")
            st.pyplot(plt.gcf())

        
        else:
            st.write("Please select both target and feature variables.")
          
