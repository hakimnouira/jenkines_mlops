import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
import mlflow
import mlflow.catboost

# Set the tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Mon_Experiment_ML")

def prepare_and_log_data(train_path, test_path):
    X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
    mlflow.log_artifact(train_path, "data")
    mlflow.log_artifact(test_path, "data")
    print('Data preparation complete.')
    return X_train, y_train, X_test, y_test

def train_and_log_model(X_train, y_train):
    # Train the model
    model = train_model(X_train, y_train)
    
    # Log the model
    mlflow.catboost.log_model(model, "model")
    
    # Save the model locally
    save_model(model, 'catboost_model.pkl')
    mlflow.log_artifact('catboost_model.pkl', "model")
    
    print('Model training complete and saved.')
    return model

def evaluate_and_log_model(model, X_test, y_test):
    # Evaluate the model
    accuracy, auc, report = evaluate_model(model, X_test, y_test)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc", auc)
    
    # Log the evaluation report
    mlflow.log_text(report, "evaluation_report.txt")
    
    print(f'Accuracy: {accuracy}\nAUC: {auc}\nReport:\n{report}')
    return accuracy, auc, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='Prepare the data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--save', type=str, help='Save the trained model to a file')
    parser.add_argument('--load', type=str, help='Load a trained model from a file')
    parser.add_argument('--run', action='store_true', help='Run the model')
    args = parser.parse_args()

    train_path = 'cleaned_train11.csv'
    test_path = 'cleaned_test11.csv'

    with mlflow.start_run():
        if args.prepare:
            prepare_and_log_data(train_path, test_path)

        elif args.train:
            X_train, y_train, X_test, y_test = prepare_and_log_data(train_path, test_path)
            train_and_log_model(X_train, y_train)

        elif args.evaluate:
            X_train, y_train, X_test, y_test = prepare_and_log_data(train_path, test_path)
            model = load_model('catboost_model.pkl')
            evaluate_and_log_model(model, X_test, y_test)

        elif args.run:
            X_train, y_train, X_test, y_test = prepare_and_log_data(train_path, test_path)
            model = train_and_log_model(X_train, y_train)
            evaluate_and_log_model(model, X_test, y_test)
            save_model(model, 'catboost_model.pkl')
            loaded_model = load_model('catboost_model.pkl')
            print('Model successfully loaded.')

        if args.save:
            model = load_model('catboost_model.pkl')
            save_model(model, args.save)
            print(f'Model saved as {args.save}')

        if args.load:
            model = load_model(args.load)
            print(f'Model {args.load} loaded successfully.')

if __name__ == '__main__':
    main()
