# UK APP Scam & Money Mule Risk Model

A transaction-risk model for Authorised Push Payment (APP) scams and money mule activity, built on synthetic UK banking data, with SHAP explanations and a Streamlit dashboard.

## Results

The first version looked excellent, but a one-line rule did just as well. The synthetic data made scam payments large and genuine payments small, so the model had learned payment size rather than fraud. I rebuilt the data and compared the model against that rule again.

| | Original data | Rebuilt data |
|---|---|---|
| Transactions | 10,000 | 120,000 |
| Fraud transactions | 48 | 507 |
| Fraud in the test set | 28 | 102 (54 investment scam, 48 money mule) |
| Rule "amount ≥ £2,000": precision on test | 100% | 5.5% |
| Rule "amount ≥ £2,000": recall on test | 96% | 32% |
| Amount as a score: PR-AUC | 0.971 | 0.052 |
| XGBoost: PR-AUC | 0.956 | **0.678** |
| XGBoost: recall in the riskiest 1% of payments | 71% (the maximum possible) | **75%** |

On the original data the model added nothing over a simple rule. On the rebuilt data the rule raises about 600 alerts and catches a third of the frauds, while the model catches three quarters with 240 alerts.

Full outputs: `evaluation/audit_results_v1_original.json` (original), `evaluation/audit_results_v2_realistic.json` (an intermediate version) and `evaluation/audit_results.json` (current). Reproduce with `python evaluation/audit.py`.

## What changed in the data

- Genuine customers make about 40 payments a year, including occasional large payments (£2,000–£15,000) and short bursts of activity.
- Investment scams start with a small payment and grow, spaced 1–7 days apart with random timing.
- Money mule transfers are £200–£3,000, with irregular gaps of 5–180 minutes.
- Many more scam scenarios, so the test set has over 100 frauds of both types.
- The model and features were not changed.

## What it does

1. **Generates synthetic banking data**: customers, accounts, devices, beneficiaries and linked transactions, with injected investment scam and money mule scenarios.
2. **Engineers behavioural features**: transaction frequency, value and velocity over time windows, new-beneficiary and device signals, and distance between transaction locations.
3. **Trains an XGBoost model** with a chronological train/test split (training on earlier transactions and testing on later ones) and class weighting for the rare fraud label.
4. **Explains predictions with SHAP**: overall feature importance, plus the top contributing features for each flagged transaction.
5. **Shows results in a Streamlit dashboard** (three pages).

## Project structure

```
src/
  config/               UK cities, occupations, risk profiles
  data_generation/      customers, accounts, devices, beneficiaries, transactions, fraud scenarios
  feature_engineering/  behavioural features
  models/               XGBoost risk model and evaluation
  explainability/       SHAP explainer
  dashboard/            Streamlit app
evaluation/             audit script and saved results
```

## Run it

```
pip install -r requirements.txt
python evaluation/audit.py
python -m streamlit run src/dashboard/app.py
```

Run from the repository root. Building the full dataset takes several minutes.

## Limitations

- All data is synthetic. The results show the evaluation method works, not how the model would perform on real bank data.
- Investment scam payments are not always sent to a newly added payee, which is a strong signal in real APP fraud.
- Transaction amount is still a strong signal (single-feature AUC 0.95). It ranks well but produces many false alarms because fraud is rare.
- The test set has 102 frauds, so the metrics carry real uncertainty.
- There are no automated tests or CI yet.

## Roadmap

- [x] Audit features and compare against a simple rule baseline
- [x] More realistic data with overlapping scam and genuine behaviour
- [x] Metrics that matter for investigators: PR-AUC and recall at a fixed alert rate
- [ ] Results by scam type
- [ ] Send investment scams to newly added payees
- [ ] Save the generated data so the dashboard loads quickly
- [ ] Tests and GitHub Actions CI
- [ ] Mule network features (graph analysis)
