
# 1) DATA LOADING & INITIAL EXPLORATION

# -----------------------
# Core & scientific stack
# -----------------------
import os
import sys
import time
from typing import List, Optional
from contextlib import contextmanager
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Modeling libraries
# -----------------------
import xgboost as xgb 

# Utilities
from joblib import Parallel, delayed
from tqdm.auto import tqdm  # notebook/console-friendly tqdm

# -----------------------
# Notebook / plotting opts
# -----------------------
%matplotlib inline
plt.rcParams["figure.figsize"] = (10, 5)
sns.set_style("whitegrid")

# -----------------------
# Paths & global config
# -----------------------
COMP_DIR = os.path.expanduser("/Users/nimish/optiver-realized-volatility-prediction")

# Safety epsilon for any division (e.g., RMSPE, sample weights)
EPS = 1e-9



# KNN/NN feature settings (kept for later sections if you use them)
ENSEMBLE_METHOD = 'mean'    # (unused in final XGB flow; preserved for compatibility)
NN_NUM_MODELS = 5           # (unused for final prediction)
TABNET_NUM_MODELS = 1       # (unused for final prediction)
SHORTCUT_NN_IN_1ST_STAGE = True
GBDT_NUM_MODELS = 0         # not used
GBDT_LR = 0.02              # not used

# -----------------------
# Timing helper
# -----------------------
@contextmanager
def timer(name: str):
    """Lightweight timing context for profiling blocks."""
    s = time.time()
    yield
    print(f"[{name}] {time.time() - s: .3f} sec")

# -----------------------
# DataBlock enum & loaders (signature-compatible with your FE code)
# -----------------------
class DataBlock(Enum):
    TRAIN = 1
    TEST  = 2
    BOTH  = 3

def load_stock_parquet(stock_id: int, directory: str) -> pd.DataFrame:
    """Load a single stock's parquet shard from book_/trade_ directories."""
    path = os.path.join(COMP_DIR, directory, f"stock_id={stock_id}")
    return pd.read_parquet(path)

def load_data(stock_id: int, stem: str, block: DataBlock) -> pd.DataFrame:
    """Load book or trade data for a given stock_id and split."""
    if block == DataBlock.TRAIN:
        return load_stock_parquet(stock_id, f"{stem}_train.parquet")
    elif block == DataBlock.TEST:
        return load_stock_parquet(stock_id, f"{stem}_test.parquet")
    else:
        # BOTH: concat train+test for utilities where that’s appropriate
        return pd.concat(
            [load_data(stock_id, stem, DataBlock.TRAIN),
             load_data(stock_id, stem, DataBlock.TEST)],
            axis=0, ignore_index=True
        )

def load_book(stock_id: int, block: DataBlock = DataBlock.TRAIN) -> pd.DataFrame:
    return load_data(stock_id, "book", block)

def load_trade(stock_id: int, block: DataBlock = DataBlock.TRAIN) -> pd.DataFrame:
    return load_data(stock_id, "trade", block)

# -----------------------
# Index files (train/test)
# -----------------------
with timer("load train/test index"):
    train = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))   # columns: stock_id, time_id, target
    test  = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))    # columns: row_id, stock_id, time_id

# Basic exploration (non-destructive)
print(f"train shape: {train.shape} | test shape: {test.shape}")
print(f"# unique stocks (train): {train['stock_id'].nunique()} | time_ids: {train['time_id'].nunique()}")
if len(test) == 3:
    IS_1ST_STAGE = True
    print("Detected 1st-stage (small) test set.")

# Keep a quick reference set for FE loops
stock_ids = set(train["stock_id"])
print(f"Loaded {len(stock_ids)} unique stock_ids.")


# ================================================================
# 2) FEATURE ENGINEERING & PREPROCESSING
#    What this section does:
#      • Builds microstructure features from order book (top-2 levels) & trades
#      • Aggregates per time_id with multiple stats and late-window cuts (150/300/450)
#      • Derives a robust tick size proxy per (stock_id, time_id)
#      • Assembles train/test featureframes for downstream modeling
#    Why these features:
#      • Volatility of weighted-average-price (WAP) log-returns captures micro-moves
#      • Spreads, depth, and imbalance summarize liquidity/pressure
#      • Late-window aggregates approximate end-of-bucket dynamics
# ================================================================

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# -----------------------
# Core microstructure helpers
# -----------------------
def calc_wap1(df: pd.DataFrame) -> pd.Series:
    """Weighted average price using L1 bid/ask with opposite sizes as weights."""
    return (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )

def calc_wap2(df: pd.DataFrame) -> pd.Series:
    """Weighted average price using L2 bid/ask with opposite sizes as weights."""
    return (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )

def realized_volatility(x: pd.Series) -> float:
    """√(sum of squared log-returns) within a time_id bucket."""
    return float(np.sqrt(np.sum(np.square(x))))

def log_return(series: pd.Series) -> pd.Series:
    """First difference of log price; aligns with microstructure volatility."""
    return np.log(series).diff()

def flatten_name(prefix: str, cols) -> list:
    """
    Convert a multiindex from groupby.agg() into flat, unique names.
    Keeps time_id/stock_id as-is for deterministic merging.
    """
    out = []
    for c in cols:
        k = c[0]
        if k in ("time_id", "stock_id"):
            out.append(k)
        else:
            out.append(".".join([prefix] + list(c)))
    return out

# -----------------------
# Book-derived features (per stock_id)
# -----------------------
def make_book_feature(stock_id, block=DataBlock.TRAIN):
    book = load_book(stock_id, block).copy()

    # Base WAPs
    book['wap1'] = calc_wap1(book)
    book['wap2'] = calc_wap2(book)

    
    book['log_return1']     = book.groupby('time_id')['wap1'].apply(log_return)
    book['log_return2']     = book.groupby('time_id')['wap2'].apply(log_return)
    book['log_return_ask1'] = book.groupby('time_id')['ask_price1'].apply(log_return)
    book['log_return_ask2'] = book.groupby('time_id')['ask_price2'].apply(log_return)
    book['log_return_bid1'] = book.groupby('time_id')['bid_price1'].apply(log_return)
    book['log_return_bid2'] = book.groupby('time_id')['bid_price2'].apply(log_return)

    # Microstructure extras
    book['wap_balance']  = (book['wap1'] - book['wap2']).abs()
    book['price_spread'] = (book['ask_price1'] - book['bid_price1']) / (
        (book['ask_price1'] + book['bid_price1']) / 2.0
    )
    book['bid_spread']   = book['bid_price1'] - book['bid_price2']
    book['ask_spread']   = book['ask_price1'] - book['ask_price2']
    book['total_volume'] = (book['ask_size1'] + book['ask_size2']) + (book['bid_size1'] + book['bid_size2'])
    book['volume_imbalance'] = (
        (book['ask_size1'] + book['ask_size2']) - (book['bid_size1'] + book['bid_size2'])
    ).abs()

    # Aggregations (kept identical to your original)
    features = {
        'seconds_in_bucket': ['count'],
        'wap1': [np.sum, np.mean, np.std],
        'wap2': [np.sum, np.mean, np.std],
        'log_return1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return2': [np.sum, realized_volatility, np.mean, np.std],
        'log_return_ask1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return_ask2': [np.sum, realized_volatility, np.mean, np.std],
        'log_return_bid1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return_bid2': [np.sum, realized_volatility, np.mean, np.std],
        'wap_balance': [np.sum, np.mean, np.std],
        'price_spread': [np.sum, np.mean, np.std],
        'bid_spread': [np.sum, np.mean, np.std],
        'ask_spread': [np.sum, np.mean, np.std],
        'total_volume': [np.sum, np.mean, np.std],
        'volume_imbalance': [np.sum, np.mean, np.std],
    }

    agg = book.groupby('time_id').agg(features).reset_index(drop=False)
    agg.columns = flatten_name('book', agg.columns)
    agg['stock_id'] = stock_id

    # Late-window cuts (identical)
    for time in [450, 300, 150]:
        d = book[book['seconds_in_bucket'] >= time].groupby('time_id').agg(features).reset_index(drop=False)
        d.columns = flatten_name(f'book_{time}', d.columns)
        agg = pd.merge(agg, d, on='time_id', how='left')

    return agg


# -----------------------
# Trade-derived features (per stock_id)
# -----------------------
def make_trade_feature(stock_id: int, block) -> pd.DataFrame:
    """
    Per-(time_id, stock_id) trades:
      • Log-return RV
      • Counts and sizes
      • Late-window cuts at 150/300/450s
    """
    trade = load_trade(stock_id, block).copy()
    trade["log_return"] = trade.groupby("time_id")["price"].apply(log_return)

    agg_spec = {
        "log_return":            [realized_volatility],
        "seconds_in_bucket":     ["count"],
        "size":                  [np.sum],
        "order_count":           [np.mean],
    }

    agg = trade.groupby("time_id").agg(agg_spec).reset_index()
    agg.columns = flatten_name("trade", agg.columns)
    agg["stock_id"] = stock_id

    for t in (450, 300, 150):
        d = trade[trade["seconds_in_bucket"] >= t].groupby("time_id").agg(agg_spec).reset_index(drop=False)
        d.columns = flatten_name(f"trade_{t}", d.columns)
        agg = pd.merge(agg, d, on="time_id", how="left")

    return agg

# -----------------------
# Tick size proxy (robust)
# -----------------------
def _tick_from_prices(values: np.ndarray) -> float:
    """
    Estimate minimal positive increment among unique prices.
    Returns NaN if insufficient distinct levels.
    """
    uniq = np.unique(values.astype(float))
    if uniq.size < 2:
        return np.nan
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    return float(np.min(diffs)) if diffs.size else np.nan

def make_book_feature_v2(stock_id: int, block) -> pd.DataFrame:
    """
    Derive per-(time_id, stock_id) tick_size estimate from L1/L2 prices.
    Useful for scaling: real_price = 0.01 / tick_size (when tick is quoted in cents).
    """
    book = load_book(stock_id, block)[["time_id", "bid_price1", "ask_price1", "bid_price2", "ask_price2"]].copy()
    prices = book.set_index("time_id")

    ticks = {}
    for tid, chunk in prices.groupby(level=0):
        vals = chunk.values.flatten()
        ticks[tid] = _tick_from_prices(vals)

    out = pd.DataFrame({"time_id": np.unique(book["time_id"])})
    out["stock_id"]  = stock_id
    out["tick_size"] = out["time_id"].map(ticks)

    # Impute per stock with median if missing
    out["tick_size"] = out.groupby("stock_id")["tick_size"].transform(lambda s: s.fillna(s.median()))
    return out

def make_trade_feature_v2(stock_id: int, block) -> pd.DataFrame:
    trade = load_trade(stock_id, block).copy()
    agg_spec = {"size": [np.mean], "order_count": [np.sum]}
    agg = trade.groupby("time_id").agg(agg_spec).reset_index()
    agg.columns = flatten_name("trade", agg.columns)
    agg["stock_id"] = stock_id

    for t in (450, 300, 150):
        d = trade[trade["seconds_in_bucket"] >= t].groupby("time_id").agg(agg_spec).reset_index(drop=False)
        d.columns = flatten_name(f"trade_{t}", d.columns)
        agg = pd.merge(agg, d, on="time_id", how="left")
    return agg

# -----------------------
# Feature assembly per split
# -----------------------
def make_features(base_index: pd.DataFrame, block, n_jobs: int = -1) -> pd.DataFrame:
    """
    Execute per-stock pipelines in parallel and merge on (stock_id, time_id).
    base_index must contain 'stock_id' and 'time_id'.
    """
    stock_ids_local = sorted(set(base_index["stock_id"]))

    books = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(make_book_feature)(sid, block) for sid in stock_ids_local
    )
    trades = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(make_trade_feature)(sid, block) for sid in stock_ids_local
    )

    book_df  = pd.concat(books, axis=0, ignore_index=True)
    trade_df = pd.concat(trades, axis=0, ignore_index=True)

    out = base_index.merge(book_df,  on=["stock_id", "time_id"], how="left")
    out = out.merge(trade_df, on=["stock_id", "time_id"], how="left")
    return out

def make_features_v2(base_index: pd.DataFrame, block, n_jobs: int = -1) -> pd.DataFrame:
    """Lightweight second-stage features (currently tick size)."""
    stock_ids_local = sorted(set(base_index["stock_id"]))
    books_v2 = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(make_book_feature_v2)(sid, block) for sid in stock_ids_local
    )
    out = pd.concat(books_v2, axis=0, ignore_index=True)
    return base_index.merge(out, on=["stock_id", "time_id"], how="left")

# -----------------------
# Build train/test featureframes
# -----------------------
FEATURES_FEATHER = os.path.join(COMP_DIR, "features_v2.f")
USE_PRECOMPUTE_FEATURES = os.path.exists(FEATURES_FEATHER)

with timer("build features (train/test)"):
    if USE_PRECOMPUTE_FEATURES:
        # Fast path: reuse cached features if present
        df = pd.read_feather(FEATURES_FEATHER)
        print("Loaded cached features:", FEATURES_FEATHER)
    else:
        # Train features
        df = make_features(train[["stock_id", "time_id", "target"]], block=DataBlock.TRAIN)
        df = make_features_v2(df, block=DataBlock.TRAIN)

        # Test features (retain row_id for later merge)
        test_df = make_features(test[["stock_id", "time_id"]], block=DataBlock.TEST)
        test_df = make_features_v2(test_df, block=DataBlock.TEST)

        # Align shapes for downstream usage
        print("train feats:", df.shape, "test feats:", test_df.shape)

        # Merge test features into a combined frame for any global transforms later
        df = pd.concat([df, test_df.merge(test[["stock_id", "time_id", "row_id"]], on=["stock_id","time_id"], how="left")],
                       axis=0, ignore_index=True)

        # Cache for reuse
        try:
            df.to_feather(FEATURES_FEATHER)
            print("Saved features to:", FEATURES_FEATHER)
        except Exception as e:
            print("WARNING: could not save features:", e)
            
            

# Light scaling / auxiliary transforms
# -----------------------
# Example: sampling time granularity proxies (counts) -> stability scalers
for base in ("trade", "book"):
    col = f"{base}.seconds_in_bucket.count"
    if col in df.columns:
        df[f"{base}.tau"] = np.sqrt(1.0 / df[col].clip(lower=1))  # √(1/N) stability proxy

# Create tau for the 150s late-window trade slice (required by NN features)
if 'trade_150.seconds_in_bucket.count' in df.columns:
    df['trade_150.tau'] = np.sqrt(1.0 / df['trade_150.seconds_in_bucket.count'].clip(lower=1))

print("Final engineered frame:", df.shape)


import gc
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import LatentDirichletAllocation

N_NEIGHBORS_MAX = 80

class Neighbors:
    def __init__(self, pivot, p, metric='minkowski', metric_params=None):
        nn = NearestNeighbors(
            n_neighbors=N_NEIGHBORS_MAX,
            p=p,
            metric=metric,
            metric_params=metric_params
        )
        nn.fit(pivot)
        self.distances, self.neighbors = nn.kneighbors(pivot, return_distance=True)

with timer('knn fit'):
    df_pv = df[['stock_id', 'time_id']].copy()
    df_pv['price'] = 0.01 / df['tick_size']
    df_pv['vol'] = df['book.log_return1.realized_volatility']
    df_pv['trade.tau'] = df['trade.tau']

    # BUG FIX: use the TRADE sum column (not book.total_volume.sum)
    df_pv['trade.size.sum'] = df['trade.size.sum']  # <-- fixed

    # --- neighbors on price pivot ---
    pivot = df_pv.pivot('time_id', 'stock_id', 'price')
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))
    k_neighbors_p2 = Neighbors(pivot, 2, metric='canberra')
    k_neighbors_p1 = Neighbors(pivot, 2, metric='mahalanobis',
                               metric_params={'V': np.cov(pivot.values.T)})
    k_neighbors_stock = Neighbors(minmax_scale(pivot.transpose()), 1)

    # --- neighbors on vol pivot ---
    pivot = df_pv.pivot('time_id', 'stock_id', 'vol')
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))
    k_neighbors_vol = Neighbors(pivot, 1)
    k_neighbors_stock_vol = Neighbors(minmax_scale(pivot.transpose()), 1)

    # --- neighbors on trade size pivot ---
    pivot = df_pv.pivot('time_id', 'stock_id', 'trade.size.sum')
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))
    k_neighbors_size = Neighbors(pivot, 2, metric='mahalanobis',
                                 metric_params={'V': np.cov(pivot.values.T)})
    k_neighbors_size_p2 = Neighbors(pivot, 2, metric='canberra')
    k_neighbors_stock_size = Neighbors(minmax_scale(pivot.transpose()), 1)

def make_neighbors(df_src, k_neighbors, feature_col, n=5):
    feature_pivot = df_src.pivot('time_id', 'stock_id', feature_col)
    feature_pivot = feature_pivot.fillna(feature_pivot.mean())
    neighbors = np.zeros((n, *feature_pivot.shape))
    for i in range(n):
        neighbors[i, :, :] += feature_pivot.values[k_neighbors[:, i], :]
    return feature_pivot, neighbors

def make_neighbors_stock(df_src, k_neighbors, feature_col, n=5):
    feature_pivot = df_src.pivot('time_id', 'stock_id', feature_col)
    feature_pivot = feature_pivot.fillna(feature_pivot.mean())
    neighbors = np.zeros((n, *feature_pivot.shape))
    for i in range(n):
        neighbors[i, :, :] += feature_pivot.values[:, k_neighbors[:, i]]
    return feature_pivot, neighbors

def make_nn_feature(df_src, neighbors, columns, index, n=5, agg=np.mean,
                    postfix='', exclude_self=False, exact=False):
    start = 1 if exclude_self else 0
    if exact:
        pivot_aggs = pd.DataFrame(neighbors[n-1, :, :], columns=columns, index=index)
    else:
        pivot_aggs = pd.DataFrame(agg(neighbors[start:n, :, :], axis=0),
                                  columns=columns, index=index)
    dst = pivot_aggs.unstack().reset_index()
    dst.columns = ['stock_id', 'time_id',
                   f'{feature_col}_cluster{n}{postfix}_{agg.__name__}']
    return dst

gc.collect()

df2 = df.copy()
print("Starting NN/Ranks from:", df2.shape)

# real_price then drop raw tick_size (it’s now encoded)
df2['real_price'] = 0.01 / df2['tick_size']
del df2['tick_size']

# Relative ranks within each time_id
df2['trade.order_count.mean'] = df2.groupby('time_id')['trade.order_count.mean'].rank()
df2['book.total_volume.sum']  = df2.groupby('time_id')['book.total_volume.sum'].rank()
df2['book.total_volume.mean'] = df2.groupby('time_id')['book.total_volume.mean'].rank()
df2['book.total_volume.std']  = df2.groupby('time_id')['book.total_volume.std'].rank()
df2['trade.tau']              = df2.groupby('time_id')['trade.tau'].rank()

for dt in [150, 300, 450]:
    df2[f'book_{dt}.total_volume.sum']  = df2.groupby('time_id')[f'book_{dt}.total_volume.sum'].rank()
    df2[f'book_{dt}.total_volume.mean'] = df2.groupby('time_id')[f'book_{dt}.total_volume.mean'].rank()
    df2[f'book_{dt}.total_volume.std']  = df2.groupby('time_id')[f'book_{dt}.total_volume.std'].rank()
    df2[f'trade_{dt}.order_count.mean'] = df2.groupby('time_id')[f'trade_{dt}.order_count.mean'].rank()

feature_cols_stock = {
    'book.log_return1.realized_volatility': [np.mean, np.min, np.max, np.std],
    'trade.seconds_in_bucket.count': [np.mean],
    'trade.tau': [np.mean],
    'trade_150.tau': [np.mean],
    'book.tau': [np.mean],
    'trade.size.sum': [np.mean],
    'book.seconds_in_bucket.count': [np.mean],
}
feature_cols = {
    'book.log_return1.realized_volatility': [np.mean, np.min, np.max, np.std],
    'real_price': [np.max, np.mean, np.min],
    'trade.seconds_in_bucket.count': [np.mean],
    'trade.tau': [np.mean],
    'trade.size.sum': [np.mean],
    'book.seconds_in_bucket.count': [np.mean],
    'trade_150.tau_cluster20_sv_mean': [np.mean],
    'trade.size.sum_cluster20_sv_mean': [np.mean],
}
time_id_neigbor_sizes     = [3, 5, 10, 20, 40]
time_id_neigbor_sizes_vol = [2, 3, 5, 10, 20, 40]
stock_id_neighbor_sizes   = [10, 20, 40]

ndf = None
def _add_ndf(ndf_local, dst):
    if ndf_local is None:
        return dst
    else:
        ndf_local[dst.columns[-1]] = dst[dst.columns[-1]].astype(np.float32)
        return ndf_local

# Across STOCKS
for feature_col in feature_cols_stock.keys():
    feature_pivot, neighbors_stock      = make_neighbors_stock(df2, k_neighbors_stock.neighbors, feature_col, n=N_NEIGHBORS_MAX)
    _,              neighbors_stock_vol = make_neighbors_stock(df2, k_neighbors_stock_vol.neighbors, feature_col, n=N_NEIGHBORS_MAX)
    _,              neighbors_stock_sz  = make_neighbors_stock(df2, k_neighbors_stock_size.neighbors, feature_col, n=N_NEIGHBORS_MAX)

    columns = feature_pivot.columns
    index   = feature_pivot.index

    for agg in feature_cols_stock[feature_col]:
        for n in stock_id_neighbor_sizes:
            dst = make_nn_feature(df2, neighbors_stock, columns, index, n=n, agg=agg, postfix='_s',
                                  exclude_self=True, exact=False)
            ndf = _add_ndf(ndf, dst)

            dst = make_nn_feature(df2, neighbors_stock_vol, columns, index, n=n, agg=agg, postfix='_sv',
                                  exclude_self=True, exact=False)
            ndf = _add_ndf(ndf, dst)

df2 = pd.merge(df2, ndf, on=['time_id', 'stock_id'], how='left')
ndf = None

# Across TIME IDs
for feature_col in feature_cols.keys():
    feature_pivot, neighbors      = make_neighbors(df2, k_neighbors_p2.neighbors, feature_col, n=N_NEIGHBORS_MAX)
    _,              neighbors_p1  = make_neighbors(df2, k_neighbors_p1.neighbors, feature_col, n=N_NEIGHBORS_MAX)
    _,              neighbors_vol = make_neighbors(df2, k_neighbors_vol.neighbors, feature_col, n=N_NEIGHBORS_MAX)
    _,              neighbors_sz  = make_neighbors(df2, k_neighbors_size.neighbors, feature_col, n=N_NEIGHBORS_MAX)
    _,              neighbors_sz2 = make_neighbors(df2, k_neighbors_size_p2.neighbors, feature_col, n=N_NEIGHBORS_MAX)

    columns = feature_pivot.columns
    index   = feature_pivot.index
    time_id_ns = time_id_neigbor_sizes_vol if 'volatility' in feature_col else time_id_neigbor_sizes

    for agg in feature_cols[feature_col]:
        for n in time_id_ns:
            for postfix, neigh, excl in [
                ('_p2',   neighbors,     True),
                ('_p1',   neighbors_p1,  False),
                ('_v',    neighbors_vol, False),
                ('_size', neighbors_sz,  False),
                ('_size_p2', neighbors_sz2, False),
            ]:
                dst = make_nn_feature(df2, neigh, columns, index, n=n, agg=agg,
                                      postfix=postfix, exclude_self=excl, exact=False)
                ndf = _add_ndf(ndf, dst)

df2 = pd.merge(df2, ndf, on=['time_id', 'stock_id'], how='left')

# Relative rank features from neighbor aggregates
for sz in time_id_neigbor_sizes:
    df2[f'real_price_rankmin_{sz}']  = df2['real_price'] / df2[f"real_price_cluster{sz}_p2_amin"]
    df2[f'real_price_rankmax_{sz}']  = df2['real_price'] / df2[f"real_price_cluster{sz}_p2_amax"]
    df2[f'real_price_rankmean_{sz}'] = df2['real_price'] / df2[f"real_price_cluster{sz}_p2_mean"]

for sz in time_id_neigbor_sizes_vol:
    df2[f'vol_rankmin_{sz}'] = df2['book.log_return1.realized_volatility'] / df2[f"book.log_return1.realized_volatility_cluster{sz}_p2_amin"]
    df2[f'vol_rankmax_{sz}'] = df2['book.log_return1.realized_volatility'] / df2[f"book.log_return1.realized_volatility_cluster{sz}_p2_amax"]

# Drop raw real_price columns (keep only relative ranks)
price_cols = [c for c in df2.columns if 'real_price' in c and 'rank' not in c]
for c in price_cols:
    del df2[c]

# Additional ranks on neighbor summaries
for sz in time_id_neigbor_sizes_vol:
    tgt = f'book.log_return1.realized_volatility_cluster{sz}_p1_mean'
    df2[f'{tgt}_rank'] = df2.groupby('time_id')[tgt].rank()

# Log-skew correction
for c in list(df2.columns):
    if any(x in c for x in ['trade.size.sum', 'trade_150.size.sum','trade_300.size.sum','trade_450.size.sum','volume_imbalance']):
        df2[c] = np.log(df2[c] + 1.0)

print("After NN & transforms:", df2.shape)

# Rolling RV by similar book volume
df2.sort_values(by=['stock_id', 'book.total_volume.sum'], inplace=True)
df2.reset_index(drop=True, inplace=True)
df2['realized_volatility_roll3_by_book.total_volume.mean']  = (
    df2.groupby('stock_id')['book.log_return1.realized_volatility']
       .rolling(3, center=True, min_periods=1).mean().reset_index()
       .sort_values(by=['level_1'])['book.log_return1.realized_volatility'].values
)
df2['realized_volatility_roll10_by_book.total_volume.mean'] = (
    df2.groupby('stock_id')['book.log_return1.realized_volatility']
       .rolling(10, center=True, min_periods=1).mean().reset_index()
       .sort_values(by=['level_1'])['book.log_return1.realized_volatility'].values
)

# Stock-id embedding via LDA
pivot_vol = df_pv.pivot('time_id', 'stock_id', 'vol')
lda = LatentDirichletAllocation(n_components=3, random_state=0)
stock_id_emb = pd.DataFrame(lda.fit_transform(pivot_vol.transpose()), index=pivot_vol.columns)
for i in range(stock_id_emb.shape[1]):
    df2[f'stock_id_emb{i}'] = df2['stock_id'].map(stock_id_emb[i])

# Hand control back to the main variable name used later
df = df2
del df2
gc.collect()
print("df (with NN features) ready for modeling:", df.shape)


# ================================================================
# 3) MODEL DEVELOPMENT & TRAINING
#    Goal:
#      • Construct contiguous, time-aware folds that mimic forecasting
#      • Optimize a tree-based model on RMSPE with 1/y² sample weights
#      • Produce out-of-fold predictions for honest offline validation
#    Why this setup:
#      • Time ordering reduces leakage across adjacent time_id buckets
#      • RMSPE matches the evaluation emphasis (scale-invariant)
#      • 1/y² weights stabilize training toward low-volatility regimes
# ================================================================

import os
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------- Train/Test splits from the engineered frame ----------
# df contains both train rows (with 'target') and test rows (with 'row_id')
df_train = df[df["target"].notna()].copy()
df_test  = df[df["target"].isna()].copy()

def get_X(df_src: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the feature matrix used for modeling.
    Keep all engineered numeric features; exclude label and non-predictive ids.
    """
    drop_cols = {"time_id", "target", "row_id"}  # keep stock_id as a numeric/categorical feature
    use_cols = [c for c in df_src.columns if c not in drop_cols]
    return df_src[use_cols]

# ---------- Evaluation metric & weights ----------
EPS = 1e-9

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Percentage Error with a safe denominator."""
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.sqrt(np.mean(((y_true - y_pred) / denom) ** 2)))

def rmspe_xgb(preds: np.ndarray, dmat: xgb.DMatrix):
    """Custom evaluation hook."""
    y_true = dmat.get_label()
    return "RMSPE", rmspe(y_true, preds), False  # lower is better

# ---------- ORIGINAL time_id ORDER + HARD-BORDER FOLDS ----------
# Try to load a precomputed order; otherwise fall back to your reconstruction function if present.
TIMEID_ORDER_CSV_1 = os.path.join(COMP_DIR, "optiver-time-id-ordered", "time_id_order.csv")
TIMEID_ORDER_CSV_2 = os.path.join(COMP_DIR, "time_id_order.csv")

with timer('calculate order of time-id'):
    if os.path.exists(TIMEID_ORDER_CSV_1):
        timeid_order = pd.read_csv(TIMEID_ORDER_CSV_1)
    elif os.path.exists(TIMEID_ORDER_CSV_2):
        timeid_order = pd.read_csv(TIMEID_ORDER_CSV_2)
    elif 'reconstruct_time_id_order' in globals():
        # Uses your t-SNE based reconstruction if you defined it earlier
        timeid_order = reconstruct_time_id_order()
    else:
        # Deterministic fallback (keeps behavior sane if files/functions are missing)
        timeid_order = pd.DataFrame({'time_id': np.sort(df_train['time_id'].unique())})

with timer('make folds'):
    # Map to sequential order
    timeid_order['time_id_order'] = np.arange(len(timeid_order))
    df_train['time_id_order'] = df_train['time_id'].map(timeid_order.set_index('time_id')['time_id_order'])

    # Sort exactly like your notebook
    df_train = df_train.sort_values(['time_id_order', 'stock_id']).reset_index(drop=True)

    # Recreate your original borders: width=383, end≈3830 (generalized if counts differ)
    width = 383
    end_order = int(timeid_order['time_id_order'].max()) + 1
    base_end = 3830 if end_order >= 3830 else end_order  # prefer your original cut if available

    folds_border = [base_end - width*4, base_end - width*3, base_end - width*2, base_end - width*1]
    time_id_orders = df_train['time_id_order'].values

    folds = []
    for i, border in enumerate(folds_border):
        idx_train = np.where(time_id_orders < border)[0]
        idx_valid = np.where((border <= time_id_orders) & (time_id_orders < border + width))[0]
        folds.append((idx_train, idx_valid))
        print(f"folds{i}: train={len(idx_train)}, valid={len(idx_valid)}")

    # Clean up helper column
    df_train.drop(columns=['time_id_order'], inplace=True)

# ---------- Build matrices AFTER the fold sorting ----------
X = get_X(df_train)
y = df_train["target"].astype(np.float32).values
feature_names = X.columns.tolist()  # freeze column order for reproducibility

# 1/y^2 weights (clamped) align training with the evaluation's scale invariance
w = 1.0 / (np.maximum(np.abs(y), EPS) ** 2)

# ---------- Model hyperparameters ----------
model_params = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_depth": 8,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "max_bin": 256,
    "tree_method": "hist",   # switch to "gpu_hist" if GPU is available
    "verbosity": 1,
}

NUM_BOOST_ROUND = 20_000
EARLY_STOPPING_ROUNDS = 500
VERBOSE_EVAL = 200

# ---------- Cross-validation training ----------
oof_pred = np.zeros(len(X), dtype=np.float32)
fold_best_rounds, fold_scores = [], []

with timer("CV training"):
    for k, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr, w_va = w[tr_idx], w[va_idx]

        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_names)
        dva = xgb.DMatrix(X_va, label=y_va, weight=w_va, feature_names=feature_names)

        booster = xgb.train(
            params=model_params,
            dtrain=dtr,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtr, "train"), (dva, "valid")],
            feval=rmspe_xgb,
            maximize=False,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=VERBOSE_EVAL,
        )

        best_ntree = booster.best_ntree_limit
        fold_best_rounds.append(best_ntree)

        oof_pred[va_idx] = booster.predict(dva, ntree_limit=best_ntree).astype(np.float32)
        score = rmspe(y_va, oof_pred[va_idx])
        fold_scores.append(score)
        print(f"[Fold {k}] best_ntree_limit={best_ntree}  RMSPE={score:.6f}")

# Store OOF predictions for diagnostics
pd.DataFrame({"oof_pred": oof_pred}).to_feather(os.path.join(COMP_DIR, "oof.f"))

# Keep average best-iteration for final fit
avg_best_rounds = int(np.round(np.mean(fold_best_rounds)))
print(f"OOF RMSPE: {rmspe(y, oof_pred):.6f} | avg best rounds: {avg_best_rounds}")

# ================================================================
# 4) EVALUATION & VALIDATION
#    What this section does:
#      • Computes and saves out-of-fold (OOF) diagnostics
#      • Summarizes per-fold and overall RMSPE (already printed in training)
#      • Emits quick sanity checks on weights and feature alignment
#    Why:
#      • OOF gives an honest estimate of generalization under the same
#        time-aware split used during training
# ================================================================

# Sanity checks
assert np.isfinite(w).all(), "Non-finite sample weights detected."
assert feature_names == X.columns.tolist(), "Feature order changed unexpectedly."

# Overall OOF summary frame for debugging & error analysis
oof_df = pd.DataFrame({
    "time_id": df_train["time_id"].values,
    "stock_id": df_train["stock_id"].values,
    "y_true": y,
    "y_pred": oof_pred,
})
oof_df["abs_pct_err"] = np.abs((oof_df["y_pred"] - oof_df["y_true"]) / np.maximum(np.abs(oof_df["y_true"]), EPS))
oof_df["squared_pct_err"] = oof_df["abs_pct_err"] ** 2

# Save diagnostics
oof_path = os.path.join(COMP_DIR, "oof_diagnostics.parquet")
oof_df.to_parquet(oof_path, index=False)
print(f"Saved OOF diagnostics -> {oof_path}")

# Quick aggregates (optional prints)
print("\nOOF by time_id (head):")
print(oof_df.groupby("time_id")["abs_pct_err"].mean().head())

print("\nOOF by stock_id (head):")
print(oof_df.groupby("stock_id")["abs_pct_err"].mean().head())

# Persist feature list for inference-time validation
np.save(os.path.join(COMP_DIR, "features.npy"), np.array(feature_names, dtype=object))

print(f"\nFinal OOF RMSPE: {rmspe(oof_df['y_true'].values, oof_df['y_pred'].values):.6f}")
print(f"Per-fold RMSPE: {[round(s, 6) for s in fold_scores]}")
print(f"Average best boosting rounds across folds: {int(np.round(np.mean(fold_best_rounds)))}")


# ================================================================
# 5) FINAL PREDICTIONS & SUBMISSION
#    What this section does:
#      • Trains on ALL training data using avg best rounds from CV
#      • Generates predictions on the test featureframe
#      • Builds submission with row_id alignment and saves artifacts
#    Why:
#      • Using all available data typically improves stability under shift
#      • Fixing num_boost_round from CV avoids dependence on a single holdout
# ================================================================

# Use the average best iteration from CV for a fixed-length full-data fit
avg_best_rounds = int(np.round(np.mean(fold_best_rounds)))
print(f"Using avg best rounds for final fit: {avg_best_rounds}")

# Full training matrices
X_full  = X
y_full  = y
w_full  = w
X_test  = get_X(df_test)

dtrain_full = xgb.DMatrix(X_full, label=y_full, weight=w_full, feature_names=feature_names)
dtest       = xgb.DMatrix(X_test, feature_names=feature_names)

with timer("Final training on all data"):
    final_model = xgb.train(
        params=model_params,
        dtrain=dtrain_full,
        num_boost_round=avg_best_rounds,
        verbose_eval=False,
    )

with timer("Inference on test"):
    test_pred = final_model.predict(dtest).astype(np.float32)

# Build submission in the SAME ROW ORDER as df_test -> dtest
# df_test retains 'row_id' from Section 2 when features were merged
assert "row_id" in df_test.columns, "row_id missing from df_test; re-merge from original test.csv if needed."
submission = df_test[["row_id"]].reset_index(drop=True).copy()
submission["target"] = test_pred

# Basic checks
assert submission["row_id"].isna().sum() == 0, "Found NaNs in row_id."
assert len(submission) == len(test_pred), "Submission length mismatch."

# Save submission
sub_path = os.path.join(os.getcwd(), "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"Saved submission -> {sub_path}")

# Save model & run metadata for reproducibility
final_model_path_json = os.path.join(COMP_DIR, "final_model.json")
final_model_path_bin  = os.path.join(COMP_DIR, "final_model.xgb")
final_model.save_model(final_model_path_json)   # human-readable JSON
final_model.save_model(final_model_path_bin)    # same path, binary overwrite is fine
np.save(os.path.join(COMP_DIR, "fold_best_rounds.npy"), np.array(fold_best_rounds, dtype=int))
np.save(os.path.join(COMP_DIR, "fold_scores.npy"),      np.array(fold_scores, dtype=float))

print("Artifacts saved:")
print(" •", final_model_path_json)
print(" •", final_model_path_bin)
print(" •", oof_path)
print(" •", os.path.join(COMP_DIR, "features.npy"))
print(" •", os.path.join(COMP_DIR, "fold_best_rounds.npy"))
print(" •", os.path.join(COMP_DIR, "fold_scores.npy"))
