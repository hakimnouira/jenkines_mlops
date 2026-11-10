# CatBoost MLOps CI/CD Pipeline with MLflow & Docker

An end-to-end Machine Learning Operations (MLOps) pipeline demonstrating reproducible model training, metric tracking with **MLflow**, and containerized inference orchestrated via **Docker Compose** and **Jenkins CI/CD**.

## Architecture Overview

```
[ Cleaned Dataset (CSV) ]
           │
           ▼
[ Feature Pipeline (model_pipeline.py) ]
           │
           ▼
[ Model Training (CatBoost) ] ──► [ MLflow Tracking & Model Registry (mlartifacts/) ]
           │
           ▼
[ Evaluation Report (Metrics/Loss) ]
           │
           ▼
[ Containerized API Service (app.py via Dockerfile) ]
```

## Key Components

- **Model Training (`main.py`, `model_pipeline.py`)**: Trains gradient-boosted decision trees using CatBoost with automated hyperparameter logging and data preprocessing.
- **Experiment Tracking (`mlartifacts/`)**: Tracks runs, hyperparameters, learning curves (`catboost_training.json`), and exports packaged models (`catboost_model.pkl`, `MLmodel`).
- **Inference Service (`app.py`)**: Lightweight web service exposing prediction endpoints for real-time inference.
- **Containerization (`Dockerfile`, `docker-compose.yml`)**: Self-contained runtime ensuring reproducible environments across development and CI.
- **Automation (`Makefile`)**: One-command targets for setup, training, evaluation, and container builds.

## Quick Start

### 1. Local Environment Setup
```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Pipeline via Makefile
```bash
# Run training and evaluation
make train

# Run local web server
make run
```

### 3. Run with Docker Compose
```bash
docker-compose up --build
```
