# src/trainingflow.py
"""
Metaflow flow to:
  1. Ingest raw World Happiness data (now from data/)
  2. Transform features & split train/test
  3. Hyperparameter-tune a RandomForestRegressor
  4. Log & register the best model in MLflow
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from metaflow import FlowSpec, step, Parameter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

class TrainingFlow(FlowSpec):
    # New parameter: where to load the CSV from
    data_path = Parameter(
        "data_path",
        help="Path to the world_happiness_report.csv file",
        default="data/world_happiness_train_data.csv"
    )
    cv_folds = Parameter('cv_folds',
                         help='Number of CV folds',
                         default=5)
    test_size = Parameter('test_size',
                          help='Fraction of data for testing',
                          default=0.2)
    random_seed = Parameter('random_seed',
                            help='Random seed',
                            default=42)
    experiment_name = Parameter('experiment_name',
                                help='MLflow experiment name',
                                default='WorldHappinessExperiment_v2')

    @step
    def start(self):
        """Load data into a DataFrame using self.data_path."""
        # If you need an absolute path, uncomment the next line:
        # full_path = os.path.abspath(self.data_path)
        full_path = self.data_path
        self.df = pd.read_csv(full_path)
        print(f"Loaded {len(self.df)} rows from {full_path}")
        self.next(self.transform)

    @step
    def transform(self):
        """Select features, drop rows with missing values, split X/y."""
        self.df = self.df.drop(columns=['Country', 'Year'], errors='ignore')
        self.df = self.df.dropna()
        self.y = self.df['Happiness_Score']
        self.X = self.df.drop(columns=['Happiness_Score'])
        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test) = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_seed
        )
        print(f"Train/test split: {self.X_train.shape[0]} train / {self.X_test.shape[0]} test")
        self.next(self.tune)

    @step
    def tune(self):
        """Hyperparameter tuning via GridSearchCV."""
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestRegressor(random_state=self.random_seed)
        grid = GridSearchCV(
            rf,
            param_grid,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid.fit(self.X_train, self.y_train)
        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        print(f"Best params: {self.best_params}")
        self.next(self.register)

    @step
    def register(self):
        """Log params, metrics, register model in MLflow, and promote to Production."""
        import time
        from mlflow.tracking import MlflowClient

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            # Log best hyperparameters
            mlflow.log_params(self.best_params)

            # Predict and log performance
            preds = self.best_model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, preds)
            mlflow.log_metric('mse', mse)
            print(f"Test MSE: {mse:.4f}")

            # Register the model
            model_name = "WorldHappinessModel_v2"
            result = mlflow.sklearn.log_model(
                self.best_model,
                artifact_path="model",
                registered_model_name=model_name
            )
            print(f"Registered model under name '{model_name}'")

            # Promote the model to Production
            client = MlflowClient()

            # It can take a few milliseconds for model to be available → small sleep
            time.sleep(2)

            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                version_to_promote = latest_versions[0].version
                client.transition_model_version_stage(
                    name=model_name,
                    version=version_to_promote,
                    stage="Production",
                    archive_existing_versions=True  # optional: archive old productions
                )
                print(f"Model version {version_to_promote} promoted to Production!")
            else:
                print("No model version found to promote.")
            
        self.next(self.end)

    @step
    def end(self):
        print("TrainingFlow is done. Check MLflow UI for details.")

if __name__ == '__main__':
    TrainingFlow()