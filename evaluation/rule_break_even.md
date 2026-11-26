# Is keeping the £2,000 rule worth it?

**Question:** At a 1% model alert rate, does the extra loss prevented by the £2,000 rule justify its alert cost?
**Method:** Use cached test scores; model alerts are the top 1%, rule-only means amount >= £2,000 outside those alerts, and scams use an ID if available or account + fraud type.
**Rule-only payments:** `TXN080001` £2,535.52; `TXN080003` £5,413.85; `TXN080004` £7,904.42; all are investment scams.
**Scam:** All three map to `ACC007888 + investment_scam`; its four payments total £15,993.49.
**Later value after each caught payment:** `TXN080001` £13,318.27; `TXN080003` £7,904.42; `TXN080004` £0.00.
**Prevented losses:** L_low = £15,853.79; L_high = £15,853.79, after deduplicating later payments already in L_low.
**Illustrative assumptions, not real figures:** cost = 506 x review cost + 503 x genuine-alert churn cost; cells show total cost (L_low pays?/L_high pays?).

| Review cost | Churn £0 | Churn £1 | Churn £5 | Churn £10 |
|---:|---:|---:|---:|---:|
| £2 | £1,012 (Y/Y) | £1,515 (Y/Y) | £3,527 (Y/Y) | £6,042 (Y/Y) |
| £5 | £2,530 (Y/Y) | £3,033 (Y/Y) | £5,045 (Y/Y) | £7,560 (Y/Y) |
| £10 | £5,060 (Y/Y) | £5,563 (Y/Y) | £7,575 (Y/Y) | £10,090 (Y/Y) |
| £20 | £10,120 (Y/Y) | £10,623 (Y/Y) | £12,635 (Y/Y) | £15,150 (Y/Y) |

**Conclusion:** Keeping the rule pays off in 16/16 illustrative cells under L_low and 16/16 under L_high. At the highest assumed cost, the margin is £703.79; the result is driven by one synthetic investment-scam scenario.
**Limitations:** Synthetic data; illustrative costs; deterrence is not measured; reimbursement is not modelled.
