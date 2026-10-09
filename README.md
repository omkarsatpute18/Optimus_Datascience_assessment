# Nevada Voter Turnout Prediction (November 2014)

Solution to the **Øptimus Data Science Predictive Modeling Exercise**.

**Scenario:** it is May 2014. Using a voter file of 50,000 Nevada records, predict who will vote in the November 2014 general election.

**Result:** the model predicts that **37.5%** of the 48,859 unique voters will turn out. On 20% of 2010 voters held out from training, it reached **AUC 0.92, accuracy 84.2%, precision 80.9%, recall 84.5% and F1 0.826**.

---

## Deliverables

| # | Deliverable | File |
|---|---|---|
| 1 | Predictions (CSV) | `predictions.csv` |
| 2 | Technical report (method, rationale, implementation, interpretation) | `[Voter Turnout Prediction Report.pdf]` |
| 3 | Presentation (1 cover + 5 content slides) | `[Voter Turnout Prediction Presentation.pdf]` |
| 4 | Complete code | `voter_turnout_analysis.ipynb` |

### `predictions.csv` format

| Column | Meaning |
|---|---|
| `optimus_id` | Voter identifier |
| model features | Independent variables used to predict turnout (lag features, `age_at_target`, profile fields) |
| `vote` | 1 = predicted to turn out, 0 = predicted to stay home |
| `vote_prob` | Estimated probability of turnout, between 0 and 1 |

---

## Repository structure

```
.
├── README.md
├── voter_turnout_analysis.ipynb   # full analysis, modeling and validation
├── voterfile .csv                 # original data (never modified)
├── predictions.csv                # final predictions
├── [technical report]
└── [presentation]
```

---

## How to run

1. Install Python 3.10 or newer, then the packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scipy "scikit-learn>=1.4" jupyter
   ```

2. Place the original data file in the same folder as the notebook, with the exact name used in the first `read_csv` cell.
3. Open the notebook and choose **Kernel → Restart & Run All**. Cells depend on earlier cells, so run them in order.
4. The last cells write `predictions.csv` and run the submission checks.

`random_state=42` is used throughout, so results are reproducible.

---

## Notebook outline

| Section | Content |
|---|---|
| EDA | Dimensions, data types, missing values, duplicates, voter-history and precinct analysis, demographics |
| Temporal and leakage audit | Which variables were available by May 2014 |
| Data preparation | Duplicate removal, feature selection, type fixes, lag-feature construction |
| Modeling | Model comparison, tuning, threshold selection, calibration |
| Evaluation | Cross-validation, held-out metrics, confusion matrix, feature importance |
| Prediction | 2014 scoring and `predictions.csv` |
| Validation | Checks on the submission file |

Every data modification is preceded by a comment, as the assignment requires.

---

## Method

**Model:** gradient boosting (`HistGradientBoostingClassifier`) with isotonic probability calibration. A logistic regression served as the benchmark.

**Training design.** The file has no 2014 general-election outcome, so the model is trained on the most recent comparable midterm, **2010**, using only information that existed before it. Every feature is expressed as *elections ago*, so the same rules apply when scoring 2014:

| Feature | Training (target 2010) | Scoring (target 2014) |
|---|---|---|
| `gen_lag2`, `gen_lag4`, ... `gen_lag10` | `vh08g`, `vh06g`, ... `vh00g` | `vh12g`, `vh10g`, ... `vh04g` |
| `pri_lag2`, ... `pri_lag10` | `vh08p`, ... `vh00p` | `vh12p`, ... `vh04p` |
| `prec_g_lag2`, `prec_p_lag2` | 2008 precinct turnout | 2012 precinct turnout |
| `age_at_target` | age minus 4 | age |

Other features: `n_prior_general`, `n_prior_primary`, and ten profile fields (`party`, `ethnicity`, `income`, `education`, `maritalstatus`, `dwellingtype`, `net_worth`, `home_owner_or_renter`, `cd`, `dma`).

**Selection and tuning**

- 5-fold cross-validation to compare models: gradient boosting scored AUC 0.923 against 0.912 for logistic regression.
- Random search over 25 hyperparameter candidates (125 fits).
- Decision threshold of **0.42**, chosen on out-of-fold predictions from the training data only (maximum F1).

---

## Data handling and leakage

- **Duplicates:** 1,141 exact duplicate rows were found and removed, leaving 48,859 unique voters.
- **Cutoff:** `vh14p` (the 2014 primary) is excluded because it would not exist in May 2014.
- **Target leakage:** the 2010 precinct turnout columns and the same-year primary are not used when training on 2010.
- **Exploration-only variables:** `historical_participation_count` and `age_group` were created during EDA and are not model features.
- **Sparse fields:** fields that are more than 90% missing (donor, NASCAR, dog-owner and musical-instrument fields) and `occupationindustry` were dropped.
- The original data file is never overwritten.

---

## Results

| Model (held-out 20% of 2010 voters, threshold 0.42) | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Tuned and calibrated (final) | 0.9219 | 0.8418 | 0.8091 | 0.8445 | 0.8264 |

- Predicted 2014 turnout: **37.5%** (mean `vote_prob` 0.372).
- Strongest drivers (permutation importance): voting in the general election two years earlier, age, then voting four years earlier. Profile fields add little.

---

## Limitations

- Validation is a random split within 2010. It does not prove the model transfers to 2014.
- General-election participation in the file falls from 74% (2002) to 26% (2012), so the overall 2014 rate may be overstated. Use `vote_prob` to rank voters rather than as an exact forecast.
- A 0 in the voter history can mean "did not vote" or "not yet registered"; the file cannot tell them apart.
- Only one election (2010) is used as the training analogue.

---

## Author

Omkar Satpute
