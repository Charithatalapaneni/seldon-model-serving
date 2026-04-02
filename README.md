# seldon-model-serving

Serves PyTorch classification models (image + text) using Seldon Core on Kubernetes. Includes pre/post-processing pipelines, ensemble routing, canary deployments via Helm, and Prometheus monitoring.

## What's in here

- **models/** — PyTorch wrappers (ResNet18 image classifier, LSTM sentiment classifier), model save/load with versioning, sample training script
- **serving/** — Seldon-compatible components: predictor (`predict()`), transformer (`transform_input`/`transform_output`), ensemble router (traffic split, round robin, voting)
- **gateway/** — FastAPI app for local testing: `/predict/image`, `/predict/text`, model listing, health
- **monitoring/** — Prometheus metrics + Grafana dashboard config
- **k8s/** — SeldonDeployment CRDs: single model, inference pipeline, ensemble, canary (90/10 split)
- **helm/** — Helm chart with configurable model image, canary toggle, pipeline toggle, autoscaling

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m uvicorn gateway.app:app --reload --port 8000
```

Test it:
```bash
# health
curl localhost:8000/health

# image prediction (uploads a file)
curl -X POST localhost:8000/predict/image -F "file=@photo.jpg"

# text prediction
curl -X POST localhost:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"model_type": "text", "data": ["this movie was great"]}'

# prometheus metrics
curl localhost:8000/metrics
```

## Training

The text classifier can be trained on synthetic data to produce a versioned model artifact:

```bash
python -m models.train
# saves to models/artifacts/text_classifier/<timestamp>/model.pt
```

The image classifier uses pretrained ResNet18 weights from torchvision (no training needed for basic ImageNet classification).

## Seldon Core deployment

Requires a K8s cluster with Seldon Core operator installed.

### Single model
```bash
kubectl apply -f k8s/seldon-single.yaml
```

### Inference pipeline (transformer → model → postprocessor)
```bash
kubectl apply -f k8s/seldon-pipeline.yaml
```

### Canary deployment (90% stable, 10% canary)
```bash
kubectl apply -f k8s/seldon-canary.yaml
```

### Via Helm
```bash
# basic deployment
helm install my-model helm/seldon-model/ -n ml-serving

# with canary enabled (10% to v2)
helm install my-model helm/seldon-model/ -n ml-serving \
  --set canary.enabled=true \
  --set canary.image.tag=v2 \
  --set canary.trafficPercent=10

# with pre/post processing pipeline
helm install my-model helm/seldon-model/ -n ml-serving \
  --set pipeline.enabled=true
```

## Project structure

```
models/
├── image_classifier.py     # ResNet18 wrapper
├── text_classifier.py      # LSTM sentiment model
├── utils.py                # save/load/versioning
├── train.py                # training script
└── artifacts/              # saved model weights

serving/
├── predictor.py            # Seldon predict() interface
├── transformer.py          # pre/post-processing
├── router.py               # ensemble/canary routing
└── metadata.py             # Seldon /metadata endpoint

gateway/
├── app.py                  # FastAPI server
├── schemas.py              # request/response models
└── client.py               # python client lib

monitoring/
├── metrics.py              # prometheus counters/histograms
└── grafana_dashboard.json  # grafana config

docker/
├── Dockerfile.predictor
├── Dockerfile.transformer
└── Dockerfile.router

k8s/
├── seldon-single.yaml      # basic deployment
├── seldon-pipeline.yaml    # transformer → model pipeline
├── seldon-ensemble.yaml    # multi-model ensemble
└── seldon-canary.yaml      # canary with traffic split

helm/seldon-model/
├── Chart.yaml
├── values.yaml
└── templates/
    ├── deployment.yaml
    └── canary.yaml
```

## Monitoring

`/metrics` exposes Prometheus metrics: prediction latency histograms, request counts by model type, error rates, active request gauge.

Import `monitoring/grafana_dashboard.json` into Grafana for p50/p95/p99 latency, per-model throughput, and error rate panels.
