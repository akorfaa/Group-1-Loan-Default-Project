# Group 1 — Nigeria Loan Default Predictor

# From project_description.md
## Key Challenges

- Class imbalance: only 21.78% of borrowers defaulted — use class_weight or scale_pos_weight
- Missing values: level_of_education (90% missing), bank details (25% missing)
- The most important features are likely previous loan behaviour columns

## Evaluation Metric

ROC-AUC — measures how well the model separates defaulters from repayers.
Also report Recall for the default class — catching defaulters is the priority.

## Suggested Approach

1. Handle missing values — impute categorical with mode, numerical with median
2. Encode categorical columns (bank_account_type, employment_status, education, bank_name)
3. Use scale_pos_weight or class_weight='balanced' to handle imbalance
4. Train and compare: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
5. Tune with RandomizedSearchCV scoring on recall or roc_auc
6. Lower prediction threshold to 0.3 to improve recall for defaulters

# From README.md
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
| File | What to do |
|---|---|
| src/preprocessing.py | Complete all TODO sections |
| src/train.py | Add all 5 models and complete tune_and_compare() |
| src/evaluate.py | Complete evaluate_model() and plot functions |
| src/predict.py | Complete predict_single() and predict_batch() |

# Roadmap/guide
## preprocessing.py 
Complete all TODO sections

## train.py
Add all 5 models and complete tune_and_compare()

## evaluate.py
Complete evaluate_model() and plot functions

## predict.py
Complete predict_single() and predict_batch()

# Before we start
- Claude explanations
- Read notebooks on models
- Revisit recordings if necessary
- Complete codes
- Claude explain again - why, is that all, nothing else, alternatives etc.
- Deploy it (with group)
- Create figma slides template
- Input content into slides (with group)
- Document every explanation (share with group)
- Rehearse presentation (with group)

# Group work
- Compare codes
- Upload to github
- figma slides template
- content on figma slides
- Final presentation slides - perfect
- Explanation for each code/slides
- Rehearsal