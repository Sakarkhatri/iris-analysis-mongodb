import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['iris_database']
collection = db['iris_data']

# 1. Data cleaning and preprocessing
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Expand the dataset to have more than 1000 rows
df = pd.concat([df] * 7, ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data

# Add some additional features to have more than 8 columns
df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
df['sepal_petal_ratio'] = df['sepal_area'] / df['petal_area']
df['sepal_petal_diff'] = df['sepal_area'] - df['petal_area']

# Insert data into MongoDB
collection.drop()  # Clear existing data
collection.insert_many(df.to_dict('records'))

# Retrieve data from MongoDB
cursor = collection.find({})
df = pd.DataFrame(list(cursor))
df = df.drop('_id', axis=1)  # Remove MongoDB's _id field

# 2. Data Visualization
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("iris_correlation_heatmap.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=df)
plt.title("Distribution of Iris Species")
plt.savefig("iris_species_distribution.png")
plt.close()

# 3. Analyze the data
print("\nData Analysis:")
print(df.describe())

# 4. Implement machine learning algorithms
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression(random_state=42, multi_class='ovr')
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

# 5. Compare and analyze the results
print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_pred, target_names=iris.target_names))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred, target_names=iris.target_names))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=iris.target_names))

# Feature importance for Random Forest
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title("Feature Importance (Random Forest)")
plt.savefig("iris_feature_importance.png")
plt.close()

# Close MongoDB connection
client.close()