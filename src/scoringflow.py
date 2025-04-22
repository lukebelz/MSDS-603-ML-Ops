import os
import pandas as pd
import mlflow
from metaflow import FlowSpec, step, Parameter

class ScoringFlow(FlowSpec):
    # Path to the CSV you want to score
    input_path = Parameter('input_path',
                           help='Path to new data CSV',
                           default='data/world_happiness_test_data.csv')
    # Where to write predictions
    output_path = Parameter('output_path',
                            help='Path to write predictions CSV',
                            default='predictions.csv')
    # Model registry name / stage
    model_name = Parameter('model_name',
                           help='Registered model name in MLflow',
                           default='WorldHappinessModel_v2')
    model_stage = Parameter('model_stage',
                            help='Model stage to load (e.g. Production)',
                            default='Production')

    @step
    def start(self):
        """Load new data."""
        self.df = pd.read_csv(self.input_path)
        print(f"Loaded {len(self.df)} rows from {self.input_path}")
        self.next(self.transform)

    @step
    def transform(self):
        """Drop identifiers & clean."""
        self.df = self.df.drop(columns=['Country', 'Year'], errors='ignore').dropna()
        self.X = self.df.drop(columns=['Happiness_Score'], errors='ignore')
        self.next(self.load_model)

    @step
    def load_model(self):
        """Fetch model from MLflow Registry."""
        model_uri = f"models:/{self.model_name}/{self.model_stage}"
        print(f"Loading model from '{model_uri}'")
        self.model = mlflow.sklearn.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        """Run inference."""
        self.df['Predicted_Happiness'] = self.model.predict(self.X)
        print("Predictions added to DataFrame")
        self.next(self.save)

    @step
    def save(self):
        """Write predictions to CSV."""
        self.df.to_csv(self.output_path, index=False)
        print(f"Saved predictions to {self.output_path}")
        self.next(self.end)

    @step
    def end(self):
        print("ScoringFlow complete.")

if __name__ == '__main__':
    ScoringFlow()