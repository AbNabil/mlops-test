# MLOps Demo Project

A minimal ML project built for DevOps engineers learning MLOps.

## Stack
- **Model**: scikit-learn (Iris classifier)
- **Experiment Tracking**: MLflow
- **Serving**: FastAPI
- **Containerization**: Docker + Docker Compose

## Project Structure
```
mlops-demo/
├── data/               # Dataset (auto-generated from sklearn)
├── src/
│   ├── train.py        # Model training + MLflow logging
│   └── predict.py      # Prediction logic
├── api/
│   └── main.py         # FastAPI serving endpoint
├── mlruns/             # MLflow experiment data (auto-created)
├── models/             # Saved model artifacts
├── Dockerfile.train    # Container for training
├── Dockerfile.api      # Container for serving
├── docker-compose.yml  # Orchestrates training + serving + MLflow UI
└── requirements.txt
```

## Quick Start

### Option 1: Docker Compose (recommended)
```bash
# Start MLflow UI + API
docker-compose up --build

# In a separate terminal, run training
docker-compose run trainer
```

### Option 2: Local
```bash
pip install -r requirements.txt

# Train the model
python src/train.py

# Start the API
uvicorn api.main:app --reload --port 8000
```

## Endpoints
- `GET  /health`         — health check
- `POST /predict`        — run inference
- `GET  /model/info`     — current model metadata

## MLflow UI
Open http://localhost:5000 to see experiments, runs, metrics, and registered models.

## Example Prediction Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
Expected response: `{"prediction": "setosa", "confidence": 0.97}`
