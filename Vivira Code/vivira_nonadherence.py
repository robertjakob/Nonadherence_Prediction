import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from imblearn.under_sampling import TomekLinks
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import sys
from contextlib import redirect_stdout
from io import TextIOWrapper
import subprocess

class Tee(TextIOWrapper):
    def __init__(self, file, mode='w'):
        super().__init__(open(file, mode), encoding='utf-8')
        self.terminal = sys.stdout
        sys.stdout = self
        self.file = open(file, mode)
        self.batch_output = []

    def write(self, message):
        try:
            message = str(message)
            self.terminal.write(message)
            self.batch_output.append(message)

            if len(self.batch_output) > 100:
                self.write_batch()

        except Exception as e:
            self.terminal.write(f"Error writing message: {e}\n")

    def flush(self):
        self.file.flush()

    def write_batch(self):
        try:
            self.file.write(''.join(self.batch_output))
            self.flush()
            self.batch_output.clear()
        except Exception as e:
            self.terminal.write(f"Error writing batch: {e}\n")

    def close(self):
        if self.batch_output:
            self.write_batch()
        self.file.close()
        sys.stdout = self.terminal
        super().close()

# Load dataset
df = pd.read_csv('vivira.csv')

# Adjusted target variables (NonAdhWeek2 to NonAdhWeek13)
target_variables = [f'NonAdhWeek{i}' for i in range(2, 14)]

# Create storage for results
metrics_storage = {
    'accuracy': pd.DataFrame(),
    'f1': pd.DataFrame(),
    'auc': pd.DataFrame(),
    'precision': pd.DataFrame(),
    'recall': pd.DataFrame(),
}

# Open a file and redirect stdout to both terminal and file
with Tee('ViviraNonadherence.txt'):
    # Iterate over target variables and corresponding feature sets
    for week in range(2, 14):
        # Define the correct range of features for the current week
        exercise_features = [f'exercise{j}' for j in range(1, (week-1)*7 + 1)]
        active_features = [f'active{j}' for j in range(1, (week-1)*7 + 1)]

        X = df[exercise_features + active_features]

        sqrt_transformer = FunctionTransformer(np.sqrt, validate=True)
        scaler = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ('sqrt_scale_exercise', Pipeline([
                    ('sqrt', sqrt_transformer),
                    ('scale', scaler)
                ]), exercise_features),
                ('scale_active', scaler, active_features)
            ]
        )

        # Target variable according to the logic
        target = f'NonAdhWeek{week}'

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, df[target], test_size=0.2, stratify=df[target], random_state=42
        )

        # Tomek Links undersampling
        tl = TomekLinks()
        X_res, y_res = tl.fit_resample(X_train, y_train)

        # Cross-validation and hyperparameter tuning
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(random_state=42))])

        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, 30, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__bootstrap': [True, False]
        }

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=20, cv=skf, scoring='f1', random_state=42)
        search.fit(X_res, y_res)

        # Best model evaluation
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Categorization based on Churn status
        churned_users = df['LastWeek'] < week
        churning_users = df['LastWeek'] == week
        active_users = df['LastWeek'] > week

        tp_churned = ((y_test == 1) & (y_pred == 1) & churned_users).sum()
        tp_churning = ((y_test == 1) & (y_pred == 1) & churning_users).sum()
        tp_active = ((y_test == 1) & (y_pred == 1) & active_users).sum()

        fp_churned = ((y_test == 0) & (y_pred == 1) & churned_users).sum()
        fp_churning = ((y_test == 0) & (y_pred == 1) & churning_users).sum()
        fp_active = ((y_test == 0) & (y_pred == 1) & active_users).sum()

        fn_churned = ((y_test == 1) & (y_pred == 0) & churned_users).sum()
        fn_churning = ((y_test == 1) & (y_pred == 0) & churning_users).sum()
        fn_active = ((y_test == 1) & (y_pred == 0) & active_users).sum()

        tn_churned = ((y_test == 0) & (y_pred == 0) & churned_users).sum()
        tn_churning = ((y_test == 0) & (y_pred == 0) & churning_users).sum()
        tn_active = ((y_test == 0) & (y_pred == 0) & active_users).sum()

        # Store results
        metrics_storage['accuracy'].loc[f'Week{week}', target] = accuracy
        metrics_storage['f1'].loc[f'Week{week}', target] = f1
        metrics_storage['auc'].loc[f'Week{week}', target] = auc
        metrics_storage['precision'].loc[f'Week{week}', target] = precision
        metrics_storage['recall'].loc[f'Week{week}', target] = recall

        # Print results
        print(f"Results for {target} with feature set up to week {week-1}:")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Best Params: {search.best_params_}\n")

        print("Confusion Matrix:")
        print(cm)
        print("\n")

        # Print additional required metrics
        print(f"True Positives (TP) - Total: {cm[1, 1]}")
        print(f"True Positives (TP) - Churned: {tp_churned}")
        print(f"True Positives (TP) - Churning: {tp_churning}")
        print(f"True Positives (TP) - Active: {tp_active}\n")

        print(f"False Positives (FP) - Total: {cm[0, 1]}")
        print(f"False Positives (FP) - Churned: {fp_churned}")
        print(f"False Positives (FP) - Churning: {fp_churning}")
        print(f"False Positives (FP) - Active: {fp_active}\n")

        print(f"False Negatives (FN) - Total: {cm[1, 0]}")
        print(f"False Negatives (FN) - Churned: {fn_churned}")
        print(f"False Negatives (FN) - Churning: {fn_churning}")
        print(f"False Negatives (FN) - Active: {fn_active}\n")

        print(f"True Negatives (TN) - Total: {cm[0, 0]}")
        print(f"True Negatives (TN) - Churned: {tn_churned}")
        print(f"True Negatives (TN) - Churning: {tn_churning}")
        print(f"True Negatives (TN) - Active: {tn_active}\n")

        # Print feature importances
        feature_importances = best_model.named_steps['classifier'].feature_importances_
        feature_names = exercise_features + active_features
        print("Feature Importances:")
        for feature_name, importance in zip(feature_names, feature_importances):
            print(f"{feature_name}: {importance:.4f}")
        print("\n")

# Save metrics to CSV files
output_dir = "ViviraNonadherence"
os.makedirs(output_dir, exist_ok=True)

for metric, df in metrics_storage.items():
    df.to_csv(os.path.join(output_dir, f'{metric}.csv'), index=True)

print("Metrics saved to CSV files.")
