import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'nigeria_loan_default_train.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

TARGET_COL   = 'target'
TEST_SIZE    = 0.2
RANDOM_STATE = 42

THRESHOLD = 0.431 # Found after tuning threshold to maximize recall for the default class (1) while maintaining reasonable precision.

DROP_COLS = ['level_of_education_clients']  # 90% missing — too sparse to use

CATEGORICAL_COLS = [
    'bank_account_type',
    'bank_name_clients',
    'employment_status_clients'
]

NUMERICAL_COLS = [
    'loannumber', 'loanamount', 'totaldue', 'termdays',
    'days_to_approve', 'interest_amount', 'interest_rate',
    'is_referred', 'approval_month', 'approval_year',
    'longitude_gps', 'latitude_gps', 'age',
    'prev_num_loans', 'prev_avg_loan_amount', 'prev_total_borrowed',
    'prev_avg_totaldue', 'prev_avg_termdays', 'prev_max_loannumber',
    'prev_avg_interest_rate', 'prev_avg_loan_duration',
    'prev_late_payments', 'prev_late_payment_rate', 'prev_avg_days_to_repay'
]

MODEL_PARAMS = {
    'n_estimators':  200,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  RANDOM_STATE,
    'verbose':       -1
}
