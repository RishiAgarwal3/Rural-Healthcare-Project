# 📊 Healthcare Call Analytics — EDA Summary Report

> **Generated:** 2026-05-27 01:25:58
> **Dataset:** enriched_flat.csv (887 calls)
> **Pipeline:** FinAI Voice Analytics

---

## 🔑 Executive Summary

| Metric | Value |
|--------|------:|
| Total Calls Analyzed | **887** |
| Identified Patients | **184** (20.7%) |
| High Urgency Cases | **55** (6.2%) |
| Unresolved Cases | **13** (1.5%) |
| Worsening Cases | **7** (0.8%) |
| Fully Recovered | **591** (66.6%) |
| Non-Compliant Patients | **17** (1.9%) |
| Follow-Up Required | **447** (50.4%) |
| Needs Manual Review | **751** (84.7%) |

---

## 🏥 Key Clinical Insights

### Patient Recovery Landscape
- **66.6%** of patients reported full recovery during follow-up calls
- **1.5%** have unresolved conditions requiring continued attention
- **0.8%** showed worsening symptoms — these are priority cases
- A large portion (**30.4%**) has unknown recovery status, indicating short or inconclusive calls

### Compliance Concerns
- Only **265** patients (29.9%) confirmed active medication compliance
- **17** patients explicitly stopped medication prematurely
- **603** calls did not yield compliance information

### Urgency Distribution
- The vast majority of calls (**93.2%**) are low urgency routine follow-ups
- **55** high-urgency cases demand immediate clinical attention
- High-urgency cases correlate strongly with unresolved/worsening conditions

---

## 🔬 NLP Findings

- Average transcript length: ~21 words (structured summary)
- Most frequently mentioned conditions: **eye**, **dental**, **blood_pressure**, **diabetes**, **thyroid**
- Primary doctor types: **general medicine**, **eye**, **dental** specialists
- Most common advice: **get_tested**, **do_not_ignore**, **continue_medication**

---

## ⚠️ Data Quality Notes

- **887** calls have missing dates (transcripts provided without timestamps)
- **863** calls have no confirmed reported conditions (most conditions appear in screening questions only)
- Extraction confidence is generally high for patient names (mean: 0.18) but lower for compliance (mean: 0.51)

---

## 📁 Output Structure

```
results/
├── charts/           — 11 PNG visualizations
├── tables/           — 15 CSV summary tables
├── reports/          — Detailed section reports
├── nlp/              — Text analysis outputs
├── correlations/     — Cross-tabulations and heatmaps
└── eda_summary.md    — This master report
```

---

## 🎯 Recommendations for Next Steps

1. **Prioritize High-Urgency Cases:** 55 cases flagged as high urgency should be routed to clinical staff immediately
2. **Address Non-Compliance:** 17 patients stopped medication — targeted outreach recommended
3. **Improve Data Collection:** 270 calls with unknown recovery suggest call scripts could be improved
4. **Enrich with Timestamps:** Adding call timestamps would enable temporal trend analysis
5. **LLM-Enhanced Extraction:** Running with LLM extraction (`--llm-extraction`) would improve accuracy for complex transcripts
