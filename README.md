# Project Overview

## Problem
Forecast next-10-minute realized volatility for hundreds of equities using high-frequency order book and trade data (1-second granularity).

## Why It Matters
Accurate short-horizon volatility forecasts strengthen option pricing, liquidity provision, and intraday risk control—enabling tighter market quotes and more resilient inventory management under fast conditions.

## Solution Approach

- **Microstructure features:** engineered spreads (L1/L2), depth, imbalance, WAP-based log-returns, realized volatility, and late-window cuts at 150s/300s/450s.  
- **Relational signals:** constructed nearest-neighbor aggregates across stocks and time windows (price, volatility, trade size pivots); added stability scalers, rolling RV measures, and stock embeddings.  
- **Modeling:** trained a single XGBoost regressor with time-ordered contiguous folds to mimic live deployment; evaluated with custom RMSPE and volatility-scaled sample weights (1/|y|²).  
- **Results:** achieved strong out-of-fold performance (OOF RMSPE = **0.19548**).

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
