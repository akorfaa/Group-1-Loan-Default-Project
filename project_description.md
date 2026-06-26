# Group 1 — Nigeria Loan Default Prediction

## The Business Problem

In Nigeria, digital lending companies approve thousands of loans every day.
Not every borrower repays. When a borrower defaults — fails to repay on time —
the lender loses the loan amount plus interest. Across millions of loans this
adds up to billions of naira in losses every year.

The challenge is that lenders cannot know in advance who will default.
They approve loans based on limited information and hope for the best.
A machine learning model that predicts default risk before loan approval
gives lenders the power to make smarter decisions — approving trustworthy
borrowers and declining high-risk ones.

## The Solution

Build a binary classification model that predicts whether a borrower will
default on their loan based on their demographic profile, current loan
details, and previous loan repayment history.

## The Business Value

- Lenders reduce losses by rejecting high-risk applicants before disbursement
- Trustworthy borrowers get faster approvals
- Financial inclusion improves as lenders gain confidence to serve more customers
- The model supports fair, data-driven lending decisions

## The Data

The dataset was originally provided by SuperLender, a Nigerian digital
lending company, through the Data Science Nigeria competition on Zindi.
It contains 4,376 loan records after merging three source tables:
demographics, loan performance, and previous loan history.


## Target Variable

| Value | Meaning |
|---|---|
| 0 | Borrower repaid the loan on time (Good) |
| 1 | Borrower defaulted — did not repay on time (Bad) |

Class distribution: 78.22% repaid (0), 21.78% defaulted (1)

## Column Descriptions

### Current Loan Features

| Column | Description |
|---|---|
| `loannumber` | Which loan number this is for the customer (e.g. 5 means their 5th loan) |
| `loanamount` | Amount of money lent in Nigerian Naira |
| `totaldue` | Total amount borrower must repay including interest and fees |
| `termdays` | Number of days the borrower has to repay the loan |
| `days_to_approve` | Days between when the customer applied and when the loan was approved |
| `interest_amount` | Total interest charged (totaldue minus loanamount) |
| `interest_rate` | Interest as a proportion of the loan amount |
| `is_referred` | 1 if the borrower was referred by another customer, 0 if not |
| `approval_month` | Month the loan was approved (1-12) |
| `approval_year` | Year the loan was approved |

### Customer Demographics

| Column | Description |
|---|---|
| `bank_account_type` | Type of primary bank account (Savings, Current, etc.) |
| `longitude_gps` | GPS longitude of the customer's location |
| `latitude_gps` | GPS latitude of the customer's location |
| `bank_name_clients` | Name of the customer's bank |
| `employment_status_clients` | Employment type (Employed, Self-Employed, Unemployed, etc.) |
| `level_of_education_clients` | Highest education level (Primary, Secondary, Tertiary, etc.) |
| `age` | Age of the customer in years (calculated from birth date) |

### Previous Loan History (Aggregated)

| Column | Description |
|---|---|
| `prev_num_loans` | Total number of loans this customer has taken before |
| `prev_avg_loan_amount` | Average loan amount across all previous loans |
| `prev_total_borrowed` | Total amount ever borrowed by this customer |
| `prev_avg_totaldue` | Average total repayment across previous loans |
| `prev_avg_termdays` | Average loan term length across previous loans |
| `prev_max_loannumber` | Highest loan number — indicates how long they have been a customer |
| `prev_avg_interest_rate` | Average interest rate paid on previous loans |
| `prev_avg_loan_duration` | Average days from approval to full repayment |
| `prev_late_payments` | Total number of previous loans where first payment was late |
| `prev_late_payment_rate` | Proportion of previous loans with a late first payment |
| `prev_avg_days_to_repay` | Average days early (negative) or late (positive) on first payments |

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
