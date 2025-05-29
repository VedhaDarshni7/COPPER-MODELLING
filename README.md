# INDUSTRIAL COPPER MODELING
### **Introduction**
  The main objective of this project is to predict the selling price using Regression Models and identify the product status using the Classification Models. Streamlit is used to enhance accessibility and usability, allowing the users to obtain predictions for selling price and product status. 
### Table of Contents
* Technologies Used
* Installation
* Import Libraries from Modules
* Data Understanding
* Data Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Model Building and Evaluation
* Model GUI
* Usage
* Features
#### Technologies Used:
* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Pickle
#### Installation
* pip install pandas
* pip install numpy
* pip install scikit-learn
* pip install matplotlib
* pip install seaborn
* pip install streamlit
* pip install pickle
##### Import Libraries from Modules
* import streamlit as st
* from streamlit_option_menu import option_menu
* import base64
* import pandas as pd
* import pickle
* from sklearn.tree import DecisionTreeClassifier
* from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
* from sklearn.model_selection import train_test_split
* from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
* import seaborn as sb
* import matplotlib.pyplot as plt
#### Data Understanding
1. Identify Variable Types: Continuous or Categorical
2. Handle Invalid Values
#### Data Preprocessing
1. Handle Missing Values: By using Mean,Median,Mode.
2. Detect Outliers: Use IQR or Isolation Forest to handle outliers.
3. Determine Skewnes: Using Log ,Box-Cox to treat skewness.
4. Encode Categorical Variables: Use One-Hot Encoding, Label Encoding, or Ordinal Encoding.
#### Exploratory Data Analysis
1. Visualize Outliers and Skewness: Visualize by using Boxplot, Distplot or Violinplot. 
2. Analyze and Treat Skewness
#### Feature Engineering
1. Create New Features: Generate through Aggregation or Transformation.
2. Drop the Correlated Features: Use heatmap and drop the highly correlated columns.
#### Model Building and Evaluation
1. Split Data
2. Model Training and Evaluate: Regression, Classification Models. Evaluate with metrics like accuracy, F1 score, Precision, Recall, Confusion Matrix.
4. Optimize with hyperparameter tuning: Use Cross-Validation and Grid Search.
#### Model GUI
1. Develop a Streamlit App for interactive predictions.
2. Allow the users to input feature values and display predictions.
#### Usage
Steps to be followed for effectively using the application:
1. Access the Streamlit App: Open the application in your browser.
2. Select target: Choose from options such as Selling Price or Product Status from the navigation menu.
3. Input Data: Enter the values for the given field.
4. Perform Prediction: By clicking the button will able to get results based on the input data.
5. Results: Prediction output will be displayed on the page.
#### Features
- **Data Preprocessing:** Handles missing values,outliers, and skewness.
- **Model Training:** Trains Machine Learning Models and hyperparameter optimization.
- **Interactive GUI:** Provides a user-friendly web interface for viewing predictions based on user input.
- **Interactive Visualizations:** Utilizes EDA techniques to understand data distributions and model performance.
- **Pickle Integration:** Saves and loads models and transformers for seamless use.

