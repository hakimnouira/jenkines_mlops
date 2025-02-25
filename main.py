import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Mon_Experiment_ML")

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
    
    if args.prepare:
        X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
        print('Data preparation complete.')
    
    elif args.train:
        X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
        model = train_model(X_train, y_train)
        save_model(model, 'catboost_model.pkl')
        mlflow.catboost.log_model(model, "model")
        print('Model training complete and saved.')
    
    elif args.evaluate:
        X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
        model = load_model('catboost_model.pkl')
        accuracy, auc, report = evaluate_model(model, X_test, y_test)
        print(f'Accuracy: {accuracy}\nAUC: {auc}\nReport:\n{report}')
    elif args.run:
         X_train, y_train, X_test, y_test = prepare_data(train_path, test_path)
         model = train_model(X_train, y_train)
         accuracy, auc, report = evaluate_model(model, X_test, y_test)
         print(f'Accuracy: {accuracy}\nAUC: {auc}\nReport:\n{report}')
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


