import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def prepare_data(train_path: str, test_path: str):
    """Loads and prepares the dataset."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.drop('Churn', axis=1)
    y_train = df_train['Churn']
    X_test = df_test.drop('Churn', axis=1)
    y_test = df_test['Churn']
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    """Trains the CatBoost model."""
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=3,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns performance metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, auc, report

def save_model(model, filename: str):
    """Saves the trained model to a file."""
    joblib.dump(model, filename)

def load_model(filename: str):
    """Loads a saved model from a file."""
    return joblib.load(filename)

