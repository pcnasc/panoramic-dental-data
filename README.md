# Panoramic Dental Data

Monorepo for the university dentistry league project: a Go backend for secure,
high-throughput processing of dental radiographs, paired with a Python module
for computer-vision / deep-learning research on the same images.

## Layout

```
.
├── go-service/        # Go backend microservice(s) — image ingestion, APIs, security
│   ├── cmd/server/     # entrypoint(s)
│   ├── internal/       # unexported application packages
│   └── go.mod
├── python-ai/          # Python AI module — CV/DL research and training
│   ├── src/dental_ai/  # importable package
│   ├── scripts/        # one-off/CLI scripts (e.g. visualize_sample.py)
│   ├── notebooks/      # exploratory notebooks
│   ├── tests/
│   └── requirements.txt
├── data/                # NEVER committed — see .gitignore
│   ├── raw/             # datasets exactly as downloaded (immutable)
│   └── processed/       # derived/cleaned data produced by python-ai
└── docs/                # architecture notes, diagrams, write-ups
```

## Data

Raw datasets live under `data/raw/<dataset-name>/` and are git-ignored —
never commit medical images or PHI. The current dataset
(`data/raw/dental-panoramic-xrays/`) has:

- `images/`, `images_cut/` — radiographs (full and cropped)
- `labels/`, `labels_cut/` — segmentation masks
- `annotations/bboxes_teeth/`, `annotations/bboxes_caries/` — bounding boxes

To re-fetch or add a dataset, drop it under `data/raw/` with a descriptive
folder name; it will be ignored by git automatically.

## Go service

```bash
cd go-service
go run ./cmd/server
```

## Python AI module

```bash
cd python-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/visualize_sample.py 360
```
