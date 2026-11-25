# Shadow-mode assessment: retire the £2,000 rule?

**Question:** If we retire the £2,000 rule, what do we lose?
**Method:** On held-out test transactions, XGBoost alerts are the top 1% by risk score; the rule is logged in shadow mode and does not create alerts.

| K=1% segment | Total frauds | Both | Rule only | Model only | Neither |
|---|---:|---:|---:|---:|---:|
| All fraud | 102 | 30 | 3 | 46 | 23 |
| investment_scam | 54 | 19 | 3 | 13 | 19 |
| money_mule_layering | 48 | 11 | 0 | 33 | 4 |

**Operational impact:** Reactivating the rule adds 506 alerts, including 3 fraud (0.59% precision).
**Conclusion:** Retiring the rule loses 3 fraud catches; keeping it adds 506 alerts at 0.59% precision.
**Limitation:** Synthetic data cannot measure deterrence (whether fraudsters avoid large payments because the rule exists).
