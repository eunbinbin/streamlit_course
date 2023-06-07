import warnings
warnings.filterwarnings('ignore')

# Import Neccessary libraries
import numpy as np
import pandas as pd
import streamlit as st

# Import Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Import Model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

#Import Sampler libraries
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline

# Set the decimal format
pd.options.display.float_format = "{:.2f}".format

df = pd.read_csv("datasets\diabetes_prediction_dataset.csv")

df.head()

# Handle duplicates
duplicate_rows_data = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_data.shape)


df = df.drop_duplicates()

# Loop through each column and count the number of distinct values
for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")

# Checking null values
print(df.isnull().sum())

# Remove Unneccessary value [0.00195%]
df = df[df['gender'] != 'Other']

df.describe().style.format("{:.2f}")

# Histogram for age
st.pyplot(plt.hist(df['age'], bins=30, edgecolor='black'))
st.pyplot(plt.title('Age Distribution'))
st.pyplot(plt.xlabel('Age'))
st.pyplot(plt.ylabel('Count'))

# Bar plot for gender
sns.countplot(x='gender', data=df)
st.pyplot(plt.title('Gender Distribution'))

# Distribution plot for BMI
sns.distplot(df['bmi'], bins=30)
st.pyplot(plt.title('BMI Distribution'))


# Count plots for binary variables
for col in ['hypertension', 'heart_disease', 'diabetes']:
    sns.countplot(x=col, data=df)
    st.pyplot(plt.title(f'{col} Distribution'))

# Count plot for smoking history
sns.countplot(x='smoking_history', data=df)
st.pyplot(plt.title('Smoking History Distribution'))

# Boxplot BMI vs Diabetes classification
sns.boxplot(x='diabetes', y='bmi', data=df)
st.pyplot(plt.title('BMI vs Diabetes'))


# Boxplot Age vs Diabetes classification
sns.boxplot(x='diabetes', y='age', data=df)
st.pyplot(plt.title('Age vs Diabetes'))

# Count plot of gender vs diabetes
sns.countplot(x='gender', hue='diabetes', data=df)
st.pyplot(plt.title('Gender vs Diabetes'))

# Boxplot HbA1c level vs Diabetes classification
sns.boxplot(x='diabetes', y='HbA1c_level', data=df)
st.pyplot(plt.title('HbA1c level vs Diabetes'))

# Boxplot blood glucose level vs Diabetes classification
sns.boxplot(x='diabetes', y='blood_glucose_level', data=df)
st.pyplot(plt.title('Blood Glucose Level vs Diabetes'))

# Pair plot for numeric features
fig = sns.pairplot(df, hue='diabetes')
st.pyplot(fig)

# Scatterplot Age vs BMI colored by Diabetes classification
sns.scatterplot(x='age', y='bmi', hue='diabetes', data=df)
st.pyplot(plt.title('Age vs BMI'))

# Violin plot of BMI against diabetes classification split by gender
sns.violinplot(x='diabetes', y='bmi', hue='gender', split=True, data=df)
st.pyplot(plt.title('BMI vs Diabetes split by Gender'))

# Interaction between gender, BMI and diabetes
sns.boxplot(x='diabetes', y='bmi', hue='gender', data=df)
st.pyplot(plt.title('BMI Distribution by Diabetes Status and Gender'))

# Interaction between gender, Age and diabetes
sns.boxplot(x='diabetes', y='age', hue='gender', data=df)
st.pyplot(plt.title('Age Distribution by Diabetes Status and Gender'))


# Define a function to map the existing categories to new ones
def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'

# Apply the function to the 'smoking_history' column
df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)

# Check the new value counts
print(df['smoking_history'].value_counts())

data = df.copy()

def perform_one_hot_encoding(df, column_name):
    # Perform one-hot encoding on the specified column
    dummies = pd.get_dummies(df[column_name], prefix=column_name)

    # Drop the original column and append the new dummy columns to the dataframe
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)

    return df

# Perform one-hot encoding on the gender variable
data = perform_one_hot_encoding(data, 'gender')

# Perform one-hot encoding on the smoking history variable
data = perform_one_hot_encoding(data, 'smoking_history')

# Compute the correlation matrix
correlation_matrix = data.corr()
#Graph I.
st.pyplot(plt.figure(figsize=(15, 10)))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
st.pyplot(plt.title("Correlation Matrix Heatmap"))


#Graph II
# Create a heatmap of the correlations with the target column
corr = data.corr()
target_corr = corr['diabetes'].drop('diabetes')

# Sort correlation values in descending order
target_corr_sorted = target_corr.sort_values(ascending=False)

sns.set(font_scale=0.8)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
st.pyplot(plt.title('Correlation with Diabetes'))

# Count plot for the 'diabetes' variable
sns.countplot(x='diabetes', data=df)
st.pyplot(plt.title('Diabetes Distribution'))

# Define resampling
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level','hypertension','heart_disease']),
        ('cat', OneHotEncoder(), ['gender','smoking_history'])
    ])

# Split data into features and target variable
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Create a pipeline that preprocesses the data, resamples data, and then trains a classifier
clf = imbPipeline(steps=[('preprocessor', preprocessor),
                      ('over', over),
                      ('under', under),
                      ('classifier', RandomForestClassifier())])

# Define the hyperparameters and the values we want to test
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}
# Create Grid Search object
grid_search = GridSearchCV(clf, param_grid, cv=5)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters: ", grid_search.best_params_)

# Convert GridSearchCV results to a DataFrame and plot
results_df = pd.DataFrame(grid_search.cv_results_)
st.pyplot(plt.figure(figsize=(8, 6)))
sns.lineplot(data=results_df, x='param_classifier__n_estimators', y='mean_test_score', hue='param_classifier__max_depth', palette='viridis')
st.pyplot(plt.title('Hyperparameters Tuning Results'))
st.pyplot(plt.xlabel('Number of Estimators'))
st.pyplot(plt.ylabel('Mean Test Score'))


# Predict on the test set using the best model
y_pred = grid_search.predict(X_test)

# Evaluate the model
print("Model Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.pyplot(plt.figure(figsize=(8, 6)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(plt.title('Confusion Matrix'))
st.pyplot(plt.xlabel('Predicted'))
st.pyplot(plt.ylabel('True'))

# After fitting the model, we input feature names
onehot_columns = list(grid_search.best_estimator_.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(['gender', 'smoking_history']))

# Then we add the numeric feature names
feature_names = ['age', 'BMI', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease'] + onehot_columns

# And now let's get the feature importances
importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_

# Create a dataframe for feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the dataframe by importance
importance_df = importance_df.sort_values('Importance', ascending=False)

# Print the feature importances
print(importance_df)

# Plot the feature importances
st.pyplot(plt.figure(figsize=(12, 8)))
sns.barplot(x='Importance', y='Feature', data=importance_df)
st.pyplot(plt.title('Feature Importances'))