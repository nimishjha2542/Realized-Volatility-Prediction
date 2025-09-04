# Project Overview

## Problem
Forecast next-10-minute realized volatility for hundreds of equities using high-frequency order book and trade data (1-second granularity).

## Why It Matters
Accurate short-horizon volatility forecasts strengthen option pricing, liquidity provision, and intraday risk control—enabling tighter market quotes and more resilient inventory management under fast conditions.

## Solution Approach

- **Microstructure features:** engineered spreads (L1/L2), depth, imbalance, WAP-based log-returns, realized volatility, and late-window cuts at 150s/300s/450s.  
- **Relational signals:** constructed nearest-neighbor aggregates across stocks and time windows (price, volatility, trade size pivots); added stability scalers, rolling RV measures, and stock embeddings.  
- **Modeling:** trained a single XGBoost regressor with time-ordered contiguous folds to mimic live deployment; evaluated with custom RMSPE and volatility-scaled sample weights (1/|y|²).  
- **Results:** • Results: **external held-out test RMSPE = 0.19548**; time-aware CV (OOF) mean **= 0.18087** across 4 folds [0.18034, 0.18072, 0.18121, 0.18121].

# Dataset

High-frequency **market microstructure data** for hundreds of equities at **1-second granularity**.  
Combines top-2 levels of the electronic order book with executed trades, enabling modeling of short-horizon (10-minute) realized volatility.

## Structure & Files
Data is stored in parquet folders partitioned by `stock_id` (e.g., `book_train.parquet/stock_id=123/...`).

### Order Book — `book_[train|test].parquet`
| column                | type   | notes                               |
|------------------------|--------|-------------------------------------|
| stock_id              | int    | Not all stocks appear in every window |
| time_id               | int    | 10-minute bucket identifier |
| seconds_in_bucket     | int    | 0–599 within each 10-minute bucket |
| bid_price1 / ask_price1 | float | normalized best bid/ask prices |
| bid_size1 / ask_size1 | float  | sizes at best bid/ask |
| bid_price2 / ask_price2 | float | normalized prices at second level |
| bid_size2 / ask_size2 | float  | sizes at second level |

### Trades — `trade_[train|test].parquet`
| column                | type   | notes                               |
|------------------------|--------|-------------------------------------|
| stock_id, time_id     | int    | Same meaning as above |
| seconds_in_bucket     | int    | May be sparse; not guaranteed to start at 0 |
| price                 | float  | Normalized VWAP within the second |
| size                  | float  | Total shares traded in that second |
| order_count           | int    | Number of unique orders executed |

### Train Labels — `train.csv`
| column   | type   | notes |
|----------|--------|-------|
| stock_id | int    | join key |
| time_id  | int    | join key |
| target   | float  | Realized volatility in the **next 10-minute window**; strictly non-overlapping with features |

### Test Mapping — `test.csv`
| column   | type   | notes |
|----------|--------|-------|
| row_id   | string | Unique row key for predictions |
| stock_id | int    | Stock identifier |
| time_id  | int    | 10-minute bucket requiring forecasts |

---

## Keys, Joins, and Indexing
- **Primary per-second index:** `(stock_id, time_id, seconds_in_bucket)` in book/trade files.  
- **Per-bucket index:** `(stock_id, time_id)` after aggregating to 10-minute buckets.  
- **Join rules:**
  - Aggregate book/trade to `(stock_id, time_id)` features.  
  - Merge with `train.csv` or `test.csv` on `(stock_id, time_id)`.  
- **Uniqueness:**
  - `row_id` is unique in test.  
  - Not every stock has data for every `time_id`.  

---

## Scale
- Hundreds of stocks  
- ~150,000 labeled training targets  
- ~3 GB total data when fully loaded

# Methodology

## 1) Data Loading & Exploration

The project begins with efficient handling of the large (~3 GB) parquet dataset.  
Each `stock_id` is stored in its own shard, enabling parallelized, on-demand access to reduce memory overhead.

### Steps & techniques
- **Custom loaders:** Wrapper functions (`load_book`, `load_trade`) fetch parquet shards by `stock_id`, with support for train/test splits.
- **Index alignment:** `train.csv` and `test.csv` establish the target set of `(stock_id, time_id)` pairs for downstream merging.
- **Profiling utilities:** A lightweight `timer` context manager tracks runtime performance during preprocessing.

## 2) Feature Engineering

Feature engineering is the core of the project, transforming raw book and trade records into informative predictors of short-horizon volatility. The design balances **microstructure intuition** with **machine learning readiness**, ensuring features are both theoretically grounded and numerically stable.

---

### A. Microstructure Features (Per-Stock, Per-Bucket)

**From order book (`book.parquet`):**
- **Weighted Average Prices (WAP1, WAP2):** price proxies from L1/L2 bid–ask levels.
- **Log-returns:** applied to WAPs and individual bid/ask levels; realized volatility computed per `time_id`.
- **Liquidity measures:** spreads (price, bid, ask), WAP balance, total volume, and imbalance across sides.
- **Late-window aggregates:** recalculated over final **150s/300s/450s** within each 10-minute bucket to capture end-of-interval dynamics.

**From trades (`trade.parquet`):**
- **Log-return volatility:** realized volatility of trade prices per bucket.
- **Activity measures:** trade counts, order counts, and total size.
- **Late-window aggregates:** same 150s/300s/450s cut logic for activity-sensitive features.

*Why:* Captures **price dynamics**, **liquidity pressure**, and **execution flow**, the three pillars of volatility formation.

---

### B. Auxiliary Features
- **Tick size proxy:** smallest positive increment among L1/L2 prices per `(stock_id, time_id)`; imputes stock-specific price grids.
- **Tau scalers:** `√(1/N)` transforms of per-bucket observation counts (`trade.tau`, `book.tau`) to stabilize variance across dense vs. sparse intervals.

*Why:* Ensures comparability across securities with heterogeneous liquidity and tick sizes.

---

### C. Relational / Cross-Sectional Features
To model dependencies **across stocks** and **across time**, relational signals are engineered using nearest-neighbor methods:
- **Pivots constructed on:** real price, realized volatility, and trade size.
- **Neighbors:** k-nearest across `time_id`s (temporal similarity) and across stocks (cross-sectional similarity).
- **Aggregates:** means, mins, maxes, stds from neighbor sets—forming “cluster features.”
- **Relative ranks:** stock’s own feature vs. neighbor aggregates (e.g., volatility rank vs. cluster mean).

*Why:* Captures market-wide regimes and **relative positioning**, critical in multi-asset volatility forecasting.

---

### D. Rolling & Embedding Features
- **Rolling realized volatility:** smoothed RV over trailing **3** and **10** buckets by stock.
- **Stock embeddings:** latent factors (e.g., via LDA) on volatility pivots, serving as low-dimensional representations of stock behavior.

*Why:* Encodes persistence and latent structure across stocks beyond hand-crafted statistics.

---

### E. Transformations & Stability Fixes
- **Log-skew correction:** applied to heavy-tailed features (e.g., trade sizes, volume imbalance).
- **Imputation:** missing late-window or sparse-bucket stats filled via stock-level medians.
- **Caching:** intermediate featureframes saved as feather/parquet to speed reruns.

*Why:* Improves training stability under heterogeneity and minimizes leakage across time windows.

---

### Output
The feature-engineering pipeline produces a **high-dimensional feature frame** per `(stock_id, time_id)`, aligned to training labels or test mappings. This dataset integrates:
- **Microstructure dynamics** (prices, spreads, imbalance)
- **Cross-sectional context** (neighbors, ranks, embeddings)
- **Temporal structure** (late-window slices, rolling volatility)

Together, these features form the backbone of the volatility forecasting model.


### Why this matters
- Shard-wise parquet loading keeps processing scalable on large datasets.
- Consistent indexing across book, trade, and labels ensures feature–target alignment with zero leakage.

  ## 3) Modeling & Training

**Objective**  
Learn stable short-horizon volatility forecasts that generalize across time and securities.

---

### A. Training Data Construction
- **Train set:** all `(stock_id, time_id)` rows from `train.csv`, with target = **next-window** realized volatility.  
- **Test set:** all `(stock_id, time_id)` pairs from `test.csv`, aligned via `row_id`.  
- **Features:** high-dimensional numeric matrix combining:
  - **Microstructure:** WAP log-returns/RV, spreads, depth, imbalance, late-window cuts (150/300/450s).
  - **Relational:** k-NN aggregates across time and stocks (price/vol/size pivots), relative ranks.
  - **Temporal:** rolling realized volatility by book volume.
  - **Stock embeddings:** low-dimensional LDA factors.
- **Exclusions:** `target`, `time_id`, and `row_id` are never used as predictors.
- **Leakage safeguards:** all features come strictly from the **current** 10-minute window; labels come from the **next** window.

---

### B. Evaluation Metric — Root Mean Squared Percentage Error (RMSPE)

**RMSPE**

$$
\mathrm{RMSPE}
= \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i-\hat{y}_i}{\max\!\left(|y_i|,\varepsilon\right)}\right)^2}
$$

- **Implementation:** denominator stabilized with $\max\\left(|y|,\varepsilon\right)$ (e.g., $\varepsilon \approx 10^{-9}$).

- **Sample weights:**

$$
w_i \=\ \frac{1}{\max\\left(|y_i|,\varepsilon\right)^2}
$$

to emphasize accuracy in low-volatility regimes where proportional errors dominate.

### C. Cross-Validation Strategy
- **Time-aware folds:** contiguous, non-overlapping splits along the ordered `time_id` sequence, mimicking live deployment.
- **Fold construction:** the full `time_id` range sorted; four validation blocks carved with hard borders (≈383 buckets each).
- **Outcome:** clean out-of-fold (OOF) predictions for unbiased diagnostics and model selection.

---

### D. Model Choice & Training
- **Model:** XGBoost regressor (tree-based GBDT) for nonlinear interactions, mixed scales, and sparse signals.
- **Key hyperparameters:**
  - `max_depth=8`, `eta=0.03`, `min_child_weight=50`
  - `subsample=0.8`, `colsample_bytree=0.6`
  - `reg_lambda=5.0`, `reg_alpha=0.0`, `max_bin=256`
  - `tree_method="hist"` (GPU-ready via `"gpu_hist"`)
- **Training regimen:** up to 20,000 boosting rounds with early stopping (patience = 500), monitored by custom RMSPE with sample weights.
- **Final fit:** retrain on all labeled data using the average “best iteration” from CV, then infer on the test feature matrix.

---

### E. Results
- **Cross-validation (4 folds, OOF):**
  - Per-fold RMSPE = `[0.18034, 0.18072, 0.18121, 0.18121]`
  - Mean CV RMSPE = `0.18087`
  - Best fold RMSPE = `0.18034`
  - Fold-to-fold variation `< 0.001` → robust generalization.
- **External held-out test:** RMSPE = **0.19548** using the final XGBoost model.
- **Interpretation:** test score is ~8.1% higher than CV mean, reflecting modest time-shift between training and held-out periods. Neighbor/rank features and volatility-scaled weights remain robust, though generalization can be tightened further under regime shifts.



