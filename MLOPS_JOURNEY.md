# MLOps Learning Journey

A practical MLOps project built from scratch. The model (Iris classifier) is just
a placeholder — the pipeline and tooling are what matter.

---

## The Full MLOps Flow

```
Code → Train → Track → Serve → Containerize → CI/CD → Deploy → Monitor
  ✅       ✅      ✅      ✅          ✅           ✅       🔄        ⬜
```

---

## What We Built

### ✅ 1. ML Model — `src/`

A Random Forest classifier trained on the Iris dataset.

- `src/train.py` — loads data, trains model, logs everything to MLflow, saves artifact
- `src/predict.py` — inference logic shared between the API and any batch jobs

Key concept: keeping training and inference code separate means you can reuse
`IrisPredictor` in a batch job, a test, or a CLI without touching the API.

```bash
# Train locally
python src/train.py

# Smoke test predictions
python src/predict.py
```

---

### ✅ 2. Experiment Tracking — MLflow

MLflow tracks every training run: hyperparameters, metrics, model artifacts.
Think of it as "git for ML experiments".

- UI at http://localhost:5000
- Each run logs: `n_estimators`, `max_depth`, `test_accuracy`, `cv_mean_accuracy`
- Model is registered in the MLflow Model Registry as `iris-rf-model`

```bash
# Start MLflow server locally
mlflow server --host 0.0.0.0 --port 5000
```

---

### ✅ 3. Model Serving — `api/`

FastAPI app that loads the trained model and exposes it as an HTTP API.

| Endpoint       | Method | Description                        |
|----------------|--------|------------------------------------|
| `/health`      | GET    | Liveness check — used by k8s probes |
| `/predict`     | POST   | Run inference on one sample        |
| `/model/info`  | GET    | Metadata about the loaded model    |
| `/docs`        | GET    | Auto-generated Swagger UI          |

```bash
# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

### ✅ 4. Containerization — Docker

Two images, one shared volume for model artifacts.

| File               | Purpose                              |
|--------------------|--------------------------------------|
| `Dockerfile.train` | Runs training job, writes model file |
| `Dockerfile.api`   | Serves the FastAPI app               |
| `docker-compose.yml` | Wires everything together locally  |

```bash
# Start everything locally
docker compose up -d

# Run training job
docker compose --profile train run trainer

# Tear down
docker compose down
```

Services:
- MLflow UI → http://localhost:5000
- API → http://localhost:8000

Healthchecks use Python's `urllib` (not curl/wget) because the slim images
don't ship those tools.

---

### ✅ 5. CI/CD — GitHub Actions

`.github/workflows/ci.yml` runs on every push to `main`:

```
push to main
    │
    ├── job: test
    │     ├── install dependencies
    │     ├── train model (local MLflow tracking)
    │     └── smoke test predictions
    │
    └── job: build  (only if test passes)
          ├── docker login (Docker Hub)
          └── build + push abnabil/mlops:latest
                                abnabil/mlops:<commit-sha>
```

Required GitHub secrets:
- `DOCKER_USERNAME` = `abnabil`
- `DOCKER_PASSWORD` = Docker Hub access token (not your password)

The commit SHA tag is important — it gives full traceability between a running
container and the exact code that produced it.

Registry: https://hub.docker.com/repository/docker/abnabil/mlops

---

### 🔄 6. Kubernetes Deployment — Helm + ArgoCD

**Status: ArgoCD installed, MTU fix applied, pending first successful sync**

#### Helm Chart — `helm/iris-api/`

```
helm/iris-api/
├── Chart.yaml
├── values.yaml          ← default values (dev)
└── templates/
    ├── deployment.yaml  ← 2 replicas, rolling update, resource limits
    ├── service.yaml     ← ClusterIP on port 80
    ├── ingress.yaml     ← nginx ingress, host: iris-api.local
    └── hpa.yaml         ← autoscale 2-5 pods at 70% CPU
```

Prod-ready features in the deployment:
- `maxUnavailable: 0` — zero downtime rolling updates
- `readinessProbe` — pod only gets traffic when healthy
- `livenessProbe` — pod restarts if it gets stuck
- CPU/memory `requests` and `limits` — prevents noisy-neighbor issues
- `imagePullSecrets` — pulls from private Docker Hub repo

```bash
# Validate chart
helm lint helm/iris-api

# Dry-run to see what would be deployed
helm template iris-api helm/iris-api --namespace mlops
```

#### ArgoCD — `argocd/application.yaml`

GitOps: ArgoCD watches the GitHub repo and syncs the cluster automatically.

```
git push (update values.yaml with new image tag)
         ↓
    ArgoCD detects change
         ↓
    helm upgrade runs in cluster
         ↓
    rolling update deploys new pods
```

Config:
- `automated.prune: true` — removes resources deleted from git
- `automated.selfHeal: true` — reverts any manual `kubectl` changes
- `CreateNamespace=true` — creates `mlops` namespace automatically

#### Cluster Setup (kind)

```bash
# ArgoCD is installed in the argocd namespace
kubectl get pods -n argocd

# Get ArgoCD admin password
kubectl get secret argocd-initial-admin-secret -n argocd \
  -o jsonpath="{.data.password}" | base64 -d

# Access ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
# open https://localhost:8080  (admin / <password above>)

# nginx ingress controller is installed
kubectl get ingressclass

# MTU fix applied to kind node (fixes git fetch timeout in ArgoCD)
docker exec kind-control-plane ip link set eth0 mtu 1400
# NOTE: this resets on kind cluster restart — re-apply if ArgoCD sync fails again
```

#### To complete the deployment

```bash
# 1. Create the mlops namespace
kubectl create namespace mlops

# 2. Create Docker Hub pull secret
kubectl create secret docker-registry dockerhub-secret \
  --namespace mlops \
  --docker-username=abnabil \
  --docker-password=<your-dockerhub-token> \
  --docker-server=https://index.docker.io/v1/

# 3. Apply the ArgoCD Application
kubectl apply -f argocd/application.yaml

# 4. Watch the sync
kubectl get application iris-api -n argocd -w

# 5. Check pods
kubectl get pods -n mlops

# 6. Add local DNS entry to test ingress
echo "127.0.0.1 iris-api.local" | sudo tee -a /etc/hosts

# 7. Test
curl http://iris-api.local/health
```

---

## ⬜ What's Next

### 7. Monitoring

Track model behavior in production — not just infrastructure metrics.

**Infrastructure (standard DevOps):**
- Request rate, latency (P95/P99), error rate
- Tool: Prometheus + Grafana

**ML-specific:**
- Prediction distribution — if one class suddenly dominates, something changed
- Input feature drift — are incoming requests different from training data?
- Confidence scores — low confidence = model is uncertain = potential drift
- Tool: Evidently (open source drift detection)

What to add:
- `prometheus-client` in the API — expose `/metrics` endpoint
- Prometheus scrape config
- Grafana dashboard with RED metrics + ML-specific panels
- Evidently report job that runs periodically

### 8. Model Registry Promotion

Currently the model goes straight from training to serving. In production you want:

```
train → staging → (manual approval or automated eval) → production
```

MLflow Model Registry already supports this with stages:
`None → Staging → Production → Archived`

What to add:
- CI step that promotes model to `Staging` after tests pass
- A promotion script that checks accuracy threshold before moving to `Production`
- ArgoCD `values-prod.yaml` that pins the image tag to the promoted version

### 9. Automated Retraining

Currently retraining is manual. In production you want it triggered automatically:

- On a schedule (weekly/monthly)
- When drift is detected (from monitoring)
- When new labeled data arrives

Tools: Airflow, Prefect, or a simple Kubernetes CronJob.

---

## Project Structure

```
mlops-test/
├── src/
│   ├── train.py              # training pipeline
│   └── predict.py            # inference logic + smoke test
├── api/
│   └── main.py               # FastAPI serving app
├── helm/
│   └── iris-api/             # Helm chart for k8s deployment
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
├── argocd/
│   └── application.yaml      # GitOps — ArgoCD watches this
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI pipeline
├── Dockerfile.api            # API container image
├── Dockerfile.train          # Training job container image
├── docker-compose.yml        # Local dev stack
├── requirements.txt
└── MLOPS_JOURNEY.md          # this file
```

## Resources

- GitHub repo: https://github.com/AbNabil/mlops-test
- Docker Hub: https://hub.docker.com/repository/docker/abnabil/mlops
- MLflow docs: https://mlflow.org/docs/latest
- ArgoCD docs: https://argo-cd.readthedocs.io
- Helm docs: https://helm.sh/docs
