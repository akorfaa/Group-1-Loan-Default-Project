# Group 1 — Nigeria Loan Default Predictor

## Setup

```bash
conda create -n loan_default python=3.10
conda activate loan_default
pip install -r requirements.txt
```

## Data

Place `nigeria_loan_default_train.csv` in the `data/` folder.

## Run

```bash
python src/train.py       # train and compare models
python src/evaluate.py    # evaluate best model
streamlit run app.py      # launch the app
```

## Files to Complete

| File                 | What to do                                       |
| -------------------- | ------------------------------------------------ |
| src/preprocessing.py | Complete all TODO sections                       |
| src/train.py         | Add all 5 models and complete tune_and_compare() |
| src/evaluate.py      | Complete evaluate_model() and plot functions     |
| src/predict.py       | Complete predict_single() and predict_batch()    |
