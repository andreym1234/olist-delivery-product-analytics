"""Load source tables, audit their grain and build one order-level SQL mart."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    'orders': 'olist_orders_dataset.csv',
    'customers': 'olist_customers_dataset.csv',
    'items': 'olist_order_items_dataset.csv',
    'payments': 'olist_order_payments_dataset.csv',
    'reviews': 'olist_order_reviews_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'category_translation': 'product_category_name_translation.csv',
}
KEYS = {
    'orders': ['order_id'], 'customers': ['customer_id'],
    'items': ['order_id', 'order_item_id'],
    'payments': ['order_id', 'payment_sequential'],
    'products': ['product_id'], 'sellers': ['seller_id'],
    'category_translation': ['product_category_name'],
}
DATE_COLUMNS = [
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date', 'review_answer_timestamp',
]


def load_tables(raw_dir: Path):
    tables, audit, manifest = {}, [], []
    for name, filename in FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing {path}. Run: python scripts/download_data.py')
        df = pd.read_csv(path)
        exact_duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
        if name in KEYS:
            key = KEYS[name]
            if df[key].isna().any().any() or df.duplicated(key).any():
                raise ValueError(f'Conflicting or missing primary key in {name}: {key}')
        # Review ids are NOT an order-level primary key. Repeated reviews are
        # retained and resolved by an explicit first-answer rule in SQL.
        tables[name] = df
        audit.append({'table': name, 'raw_rows': len(df) + exact_duplicates,
                      'exact_duplicates_removed': exact_duplicates,
                      'rows_after_exact_dedup': len(df)})
        manifest.append({'file': filename, 'bytes': path.stat().st_size,
                         'rows': len(df) + exact_duplicates,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    # Referential integrity checks: a silent failed join would change denominators.
    relations = [
        ('orders', 'customer_id', 'customers', 'customer_id'),
        ('items', 'order_id', 'orders', 'order_id'),
        ('items', 'seller_id', 'sellers', 'seller_id'),
        ('items', 'product_id', 'products', 'product_id'),
        ('payments', 'order_id', 'orders', 'order_id'),
        ('reviews', 'order_id', 'orders', 'order_id'),
    ]
    for child, ck, parent, pk in relations:
        bad = ~tables[child][ck].isin(tables[parent][pk])
        if bad.any():
            raise ValueError(f'Orphan foreign keys: {child}.{ck}: {bad.sum()}')
    return tables, pd.DataFrame(audit), manifest


def build_mart(root: Path):
    tables, audit, manifest = load_tables(root / 'data/raw')
    # Only columns required by the SQL research are loaded into SQLite.
    # Free-text reviews and unused dimensions would enlarge the database
    # without adding information to the questions in this project.
    sql_columns = {
        'orders': list(tables['orders'].columns),
        'customers': ['customer_id', 'customer_unique_id', 'customer_state'],
        'items': ['order_id', 'order_item_id', 'product_id', 'seller_id', 'price', 'freight_value'],
        'payments': ['order_id', 'payment_value'],
        'reviews': ['review_id', 'order_id', 'review_score', 'review_answer_timestamp'],
        'products': ['product_id', 'product_category_name'],
        'sellers': ['seller_id', 'seller_state'],
        'category_translation': list(tables['category_translation'].columns),
    }
    connection = sqlite3.connect(':memory:')
    try:
        for name, df in tables.items():
            df[sql_columns[name]].to_sql(name, connection, if_exists='replace', index=False)
        connection.executescript((root / 'sql/01_order_mart.sql').read_text())
        mart = pd.read_sql_query('SELECT * FROM order_mart', connection)
        sql_monthly = pd.read_sql_query(
            (root / 'sql/02_product_metrics.sql').read_text(), connection)
    finally:
        connection.close()
    assert len(mart) == len(tables['orders'])
    assert mart.order_id.is_unique
    # Validate totals after aggregation, independently of the join implementation.
    assert np.isclose(mart.goods_value.sum(), tables['items'].price.sum())
    assert np.isclose(mart.freight_value.sum(), tables['items'].freight_value.sum())
    assert np.isclose(mart.payment_value.sum(), tables['payments'].payment_value.sum())
    return mart, audit, manifest, sql_monthly


def engineer_features(mart: pd.DataFrame) -> pd.DataFrame:
    df = mart.copy()
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
    purchase = df.order_purchase_timestamp
    delivered = df.order_delivered_customer_date
    approved = df.order_approved_at
    promise = df.order_estimated_delivery_date
    df['delivery_days'] = (delivered - purchase).dt.total_seconds() / 86400
    df['late_days'] = (delivered.dt.normalize() - promise.dt.normalize()).dt.days
    df['is_late'] = df.late_days.gt(0)
    df['purchase_month'] = purchase.dt.strftime('%Y-%m')
    df['purchase_weekday'] = purchase.dt.dayofweek.astype('Int64').astype(str)
    df['promised_days_at_approval'] = (promise - approved).dt.total_seconds() / 86400
    df['order_value'] = df.goods_value + df.freight_value
    df['freight_share'] = df.freight_value / df.order_value.replace(0, np.nan)
    df['route'] = np.where(df.customer_state.eq(df.seller_state),
                           'same_state', 'cross_state')
    df.loc[df.seller_count.ne(1) | df.seller_state.isna()
           | df.customer_state.isna(), 'route'] = 'unknown_or_multi_seller'
    df['review_lag_days'] = (
        df.review_answer_timestamp - delivered).dt.total_seconds() / 86400
    df['has_review'] = df.review_score.between(1, 5)
    df['eligible_review'] = df.has_review & df.review_lag_days.between(0, 30)
    df['low_review'] = df.review_score.le(2) & df.has_review
    df['valid_delivery'] = (
        df.order_status.eq('delivered') & purchase.notna() & delivered.notna()
        & promise.notna() & df.delivery_days.ge(0)
        & df.customer_unique_id.notna() & df.item_count.gt(0)
        & df.invalid_item_count.eq(0)
        & (approved.isna() | delivered.ge(approved))
        & (df.order_delivered_carrier_date.isna()
           | delivered.ge(df.order_delivered_carrier_date))
    )
    df['cohort'] = np.select([
        purchase.ge('2017-01-01') & purchase.lt('2018-07-01'),
        purchase.ge('2018-07-01') & purchase.lt('2018-09-01'),
    ], ['discovery', 'validation'], default='outside')
    return df


def first_per_customer(df: pd.DataFrame) -> pd.DataFrame:
    """First eligible observed order, with a deterministic tie break."""
    return (df.sort_values(['order_purchase_timestamp', 'order_id'])
            .drop_duplicates('customer_unique_id', keep='first').copy())


def delivery_sample(df, cohort=None):
    mask = df.valid_delivery & df.cohort.ne('outside')
    if cohort is not None:
        mask &= df.cohort.eq(cohort)
    return df.loc[mask].copy()


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    delivered = df.order_status.eq('delivered')
    in_scope = df.cohort.ne('outside')
    measures = {
        'all_orders': len(df),
        'status_delivered': delivered.sum(),
        'delivered_invalid_or_missing_fields': (delivered & ~df.valid_delivery).sum(),
        'valid_delivered_outside_period': (df.valid_delivery & ~in_scope).sum(),
        'main_delivered_orders': (df.valid_delivery & in_scope).sum(),
        'main_without_first_review': (df.valid_delivery & in_scope & ~df.has_review).sum(),
        'main_review_before_delivery': (df.valid_delivery & in_scope & df.has_review
                                        & df.review_lag_days.lt(0)).sum(),
        'main_review_after_30_days': (df.valid_delivery & in_scope & df.has_review
                                     & df.review_lag_days.gt(30)).sum(),
        'main_review_timestamp_missing': (df.valid_delivery & in_scope & df.has_review
                                          & df.review_lag_days.isna()).sum(),
        'main_eligible_review': (df.valid_delivery & in_scope & df.eligible_review).sum(),
        'orders_with_multiple_review_records': df.review_records.gt(1).sum(),
        'orders_with_multiple_sellers': df.seller_count.gt(1).sum(),
        'payment_minus_items_freight_abs_gt_1': (
            (df.payment_value - df.order_value).abs().gt(1)).sum(),
    }
    return pd.DataFrame([{'metric': k, 'count': int(v)} for k, v in measures.items()])


def write_json(path: Path, payload):
    def convert(value):
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, (pd.Timestamp, Path)):
            return str(value)
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=convert, allow_nan=False), encoding='utf-8')
