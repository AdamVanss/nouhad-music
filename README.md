# Nouhad & Adam

Stem separator, Boy/Girl face classifier, and the Cinematch movie recommender.

A friend who already has Node and Python 3.10+ can install everything with:

```bash
bash setup.sh
```

Then, from this folder, with the virtualenv active:

```bash
python -m uvicorn src.serve:app --port 8000
```

```bash
cd web && npm run dev
```

```bash
cd movie-recommendation-system && streamlit run app.py
```

The stem and face app is at http://localhost:3000. The movie app is at http://localhost:8501.

The stem separator uses pretrained HTDemucs, which downloads on first use. The face model is `checkpoints/gender_cnn.pt`. The movie tables are downloaded by `setup.sh`.
