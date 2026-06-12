# data/

This directory holds the raw dataset CSV files and the generated FAISS vector store.

## Download the dataset

1. Visit https://www.kaggle.com/datasets/stackoverflow/pythonquestions
2. Download and unzip — you need at least:
   - `Questions.csv`
   - `Answers.csv`
3. Place both files in this `data/` directory.

## Build the vector store

```bash
python scripts/ingest_data.py \
    --questions data/Questions.csv \
    --answers   data/Answers.csv  \
    --limit     50000
```

This creates `data/vectorstore/` which the API loads on startup.

## Directory structure after ingestion

```
data/
├── Questions.csv
├── Answers.csv
└── vectorstore/
    ├── index.faiss
    └── index.pkl
```
