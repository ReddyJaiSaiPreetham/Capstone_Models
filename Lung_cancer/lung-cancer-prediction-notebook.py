# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('survey lung cancer.csv')
print("Dataset loaded successfully.")

# Display dataset information
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Check column names
print("\nActual column names in the dataset:")
for col in df.columns:
    print(f"'{col}'")

# 3. Data Preprocessing
print("\n--- Data Preprocessing ---")

# Get all column names to avoid KeyError
all_columns = list(df.columns)
target_column = 'LUNG_CANCER'  # Assuming this is the target column

# Find feature columns
feature_columns = [col for col in all_columns if col != target_column]

# Encode categorical variables
print("\nEncoding categorical variables...")
le = LabelEncoder()

# Encode each column in the dataset
for col in all_columns:
    # Check if column contains string/object data
    if df[col].dtype == 'object':
        print(f"Encoding column: '{col}'")
        df[col] = le.fit_transform(df[col])

# Display encoded dataset
print("\nEncoded dataset head:")
print(df.head())

# 4. Split features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 6. Build and train 5 ML models
print("\n--- Building and Training 5 ML Models ---")

# Function to train a model and return metrics
def train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print(f"{name} Training Accuracy: {train_accuracy:.4f}")
    print(f"{name} Testing Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print(f"\n{name} Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    return model, {
        'Model': name,
        'Training Accuracy': f"{train_accuracy:.4f}",
        'Testing Accuracy': f"{test_accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1 Score': f"{f1:.4f}"
    }

# Create 5 different models with standardization
models = {
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42))
    ]),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'AdaBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', AdaBoostClassifier(random_state=42))
    ])
}

# Train and evaluate all models
results = []
trained_models = {}

for name, model in models.items():
    trained_model, metrics = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
    results.append(metrics)
    trained_models[name] = trained_model

# 7. Create results table
results_df = pd.DataFrame(results)
print("\n--- Model Performance Comparison ---")
print(results_df.to_string(index=False))

# 8. Save the results table to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("\nResults saved to 'model_comparison_results.csv'")

# 9. Find the best model
best_model_name = results_df.loc[results_df['Testing Accuracy'].astype(float).idxmax()]['Model']
best_model = trained_models[best_model_name]

print(f"\n--- Best Model: {best_model_name} ---")

# 10. Save the best model
print("\n--- Saving the best model ---")
with open('lung_cancer_prediction_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model saved as 'lung_cancer_prediction_model.pkl'")

# 11. Visualize model comparison
plt.figure(figsize=(12, 8))
results_for_plot = results_df.copy()
results_for_plot['Testing Accuracy'] = results_for_plot['Testing Accuracy'].astype(float)
results_for_plot = results_for_plot.sort_values('Testing Accuracy', ascending=False)

sns.barplot(x='Model', y='Testing Accuracy', data=results_for_plot)
plt.title('Model Comparison - Testing Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1.0)  # Set y-axis from 0 to 1
plt.tight_layout()
plt.savefig('model_comparison.png')
print("Model comparison chart saved as 'model_comparison.png'")

# Create a heatmap of all metrics
plt.figure(figsize=(12, 8))
metrics_df = results_df.copy()
for col in ['Training Accuracy', 'Testing Accuracy', 'Precision', 'Recall', 'F1 Score']:
    metrics_df[col] = metrics_df[col].astype(float)

metrics_heatmap = metrics_df.set_index('Model')
sns.heatmap(metrics_heatmap, annot=True, cmap='Blues', fmt='.4f')
plt.title('Model Performance Metrics')
plt.tight_layout()
plt.savefig('model_metrics_heatmap.png')
print("Metrics heatmap saved as 'model_metrics_heatmap.png'")

print("\n--- Model training and evaluation complete ---")