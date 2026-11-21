# UK APP Scam & Money Mule Risk Model

A transaction-risk model for Authorised Push Payment (APP) scams and money mule activity, built on synthetic UK banking data, with SHAP explanations and a Streamlit dashboard.

> **Status: being rebuilt.** The first version scored far too well. The synthetic scam patterns were too easy to separate from normal behaviour, so the scores do not reflect real-world performance. I'm rebuilding the data and evaluation (see Roadmap). No headline metrics are reported until that is done.

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
```

## Run it

```
pip install -r requirements.txt
python -m streamlit run src/dashboard/app.py
```

Run from the repository root.

## Limitations

- All data is synthetic. It can show how the pipeline works, but not how well it would detect real fraud.
- The injected scam patterns are currently too distinct, which makes the model look much better than it is.
- There are no automated tests or CI yet.

## Roadmap

- [ ] More realistic scam data that overlaps with genuine behaviour
- [ ] Leakage audit of every feature
- [ ] A simple baseline (rules or logistic regression) to compare against
- [ ] Metrics that matter for investigators: PR-AUC and recall at a fixed alert rate
- [ ] Tests and GitHub Actions CI
- [ ] Mule network features (graph analysis)
