import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import logging
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="pipeline.log"
)

# Generate a key for encryption (store this securely in production!)
key = Fernet.generate_key()
cipher = Fernet(key)

def prepare_data(train_path: str, test_path: str):
    """
    Loads and prepares the dataset.

    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the testing dataset.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    try:
        logging.info("Loading and preparing data...")
        
        # Load datasets
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        # Validate that required columns exist
        required_columns = ['Churn']
        for column in required_columns:
            if column not in df_train.columns or column not in df_test.columns:
                raise ValueError(f"Required column '{column}' is missing in the dataset.")
        
        # Split features and target
        X_train = df_train.drop('Churn', axis=1)
        y_train = df_train['Churn']
        X_test = df_test.drop('Churn', axis=1)
        y_test = df_test['Churn']
        
        logging.info("Data preparation complete.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def train_model(X_train, y_train):
    """
    Trains the CatBoost model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        CatBoostClassifier: Trained model.
    """
    try:
        logging.info("Training the model...")
        
        # Define and train the model
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=3,
            random_state=42,
            verbose=False
        )
        model.fit(X_train, y_train)
        
        logging.info("Model training complete.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns performance metrics.

    Args:
        model (CatBoostClassifier): Trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.

    Returns:
        tuple: (accuracy, auc, report)
    """
    try:
        logging.info("Evaluating the model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logging.info(f"Model evaluation complete. Accuracy: {accuracy}, AUC: {auc}")
        return accuracy, auc, report
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_model(model, filename: str):
    """
    Saves the trained model to a file with encryption.

    Args:
        model (CatBoostClassifier): Trained model.
        filename (str): Path to save the model.
    """
    try:
        logging.info(f"Saving model to {filename}...")
        
        # Serialize the model
        model_bytes = joblib.dumps(model)
        
        # Encrypt the model
        encrypted_model = cipher.encrypt(model_bytes)
        
        # Save the encrypted model
        with open(filename, "wb") as f:
            f.write(encrypted_model)
        
        logging.info("Model saved and encrypted successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(filename: str):
    """
    Loads a saved model from an encrypted file.

    Args:
        filename (str): Path to the saved model.

    Returns:
        CatBoostClassifier: Loaded model.
    """
    try:
        logging.info(f"Loading model from {filename}...")
        
        # Read the encrypted model
        with open(filename, "rb") as f:
            encrypted_model = f.read()
        
        # Decrypt the model
        model_bytes = cipher.decrypt(encrypted_model)
        
        # Deserialize the model
        model = joblib.loads(model_bytes)
        
        logging.info("Model loaded and decrypted successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def main():
    """
    Main function to run the pipeline.
    """
    try:
        # Define file paths
        train_path = "cleaned_train11.csv"
        test_path = "cleaned_test11.csv"
        
        # Prepare data
        X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
        
        # Train the model
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        accuracy, auc, report = evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {accuracy}\nAUC: {auc}\nReport:\n{report}")
        
        # Save the model
        save_model(model, "catboost_model_encrypted.pkl")
        
        # Load the model (optional)
        loaded_model = load_model("catboost_model_encrypted.pkl")
        print("Model successfully loaded and decrypted.")
    except Exception as e:
        logging.error(f"Error in pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()
