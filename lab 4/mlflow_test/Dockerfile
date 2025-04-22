FROM python:3.10

WORKDIR /mlflow

# Install MLflow
RUN pip install mlflow

# Make sure a directory exists for tracking artifacts
RUN mkdir /mlflow/mlruns

# Expose the port MLflow UI runs on
EXPOSE 5000

# Run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "file:/mlflow/mlruns"]