"""Tests target errors that materially change product conclusions."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.data import engineer_features, first_per_customer
from src.model import prepare_model_splits
from src.statistics import ab_sample_size, compare_proportions, holm_adjust, wilson_interval


def order(**changes):
    values = {
        'order_id': 'o1', 'customer_unique_id': 'u1', 'order_status': 'delivered',
        'order_purchase_timestamp': '2018-06-01 10:00:00',
        'order_approved_at': '2018-06-01 11:00:00',
        'order_delivered_carrier_date': '2018-06-02 10:00:00',
        'order_delivered_customer_date': '2018-06-10 22:30:00',
        'order_estimated_delivery_date': '2018-06-10 00:00:00',
        'review_answer_timestamp': '2018-06-11 12:00:00',
        'review_score': 2, 'review_records': 1,
        'goods_value': 100., 'freight_value': 10., 'payment_value': 110.,
        'item_count': 1, 'seller_count': 1, 'invalid_item_count': 0,
        'customer_state': 'SP', 'seller_state': 'SP', 'main_seller_id': 's1',
        'category': 'toys',
    }
    values.update(changes)
    return values


def test_same_promised_day_is_not_late_even_in_the_evening():
    df = engineer_features(pd.DataFrame([order()]))
    assert df.loc[0, 'valid_delivery']
    assert not df.loc[0, 'is_late']
    assert df.loc[0, 'late_days'] == 0


def test_next_calendar_day_is_late():
    df = engineer_features(pd.DataFrame([
        order(order_delivered_customer_date='2018-06-11 00:00:01')]))
    assert df.loc[0, 'is_late']


def test_early_review_is_kept_but_not_in_postdelivery_primary_sample():
    df = engineer_features(pd.DataFrame([
        order(review_answer_timestamp='2018-06-09 12:00:00')]))
    assert df.loc[0, 'has_review']
    assert df.loc[0, 'low_review']
    assert df.loc[0, 'valid_delivery']
    assert not df.loc[0, 'eligible_review']


def test_impossible_delivery_order_is_excluded_without_dropping_long_deliveries():
    df = engineer_features(pd.DataFrame([
        order(order_delivered_customer_date='2018-05-31 10:00:00'),
        order(order_id='o2', order_delivered_customer_date='2018-09-01 10:00:00'),
    ]))
    assert not df.loc[0, 'valid_delivery']
    assert df.loc[1, 'valid_delivery']
    assert df.loc[1, 'delivery_days'] > 80


def test_multiseller_order_has_no_single_state_route():
    df = engineer_features(pd.DataFrame([order(seller_count=2)]))
    assert df.loc[0, 'route'] == 'unknown_or_multi_seller'


def test_sql_aggregation_does_not_multiply_items_payments_or_reviews():
    tables = {
        'orders': pd.DataFrame([{'order_id': 'o1', 'customer_id': 'c1',
                                 'order_purchase_timestamp': '2018-01-01'}]),
        'customers': pd.DataFrame([{'customer_id': 'c1', 'customer_unique_id': 'u1', 'customer_state': 'SP'}]),
        'items': pd.DataFrame([
            {'order_id': 'o1', 'order_item_id': 1, 'product_id': 'p1', 'seller_id': 's1', 'price': 10., 'freight_value': 2.},
            {'order_id': 'o1', 'order_item_id': 2, 'product_id': 'p2', 'seller_id': 's1', 'price': 20., 'freight_value': 3.},
        ]),
        'payments': pd.DataFrame([{'order_id': 'o1', 'payment_value': 7.}, {'order_id': 'o1', 'payment_value': 28.}]),
        'reviews': pd.DataFrame([
            {'order_id': 'o1', 'review_id': 'r1', 'review_score': 2, 'review_answer_timestamp': '2018-01-10'},
            {'order_id': 'o1', 'review_id': 'r2', 'review_score': 5, 'review_answer_timestamp': '2018-01-12'},
        ]),
        'products': pd.DataFrame([{'product_id': 'p1', 'product_category_name': 'a'}, {'product_id': 'p2', 'product_category_name': 'b'}]),
        'sellers': pd.DataFrame([{'seller_id': 's1', 'seller_state': 'SP'}]),
        'category_translation': pd.DataFrame([{'product_category_name': 'a', 'product_category_name_english': 'A'}, {'product_category_name': 'b', 'product_category_name_english': 'B'}]),
    }
    with sqlite3.connect(':memory:') as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
        db.executescript((Path(__file__).resolve().parents[1]/'sql/01_order_mart.sql').read_text())
        result = pd.read_sql_query('SELECT * FROM order_mart', db)
    assert len(result) == 1
    assert result.loc[0, 'goods_value'] == 30
    assert result.loc[0, 'freight_value'] == 5
    assert result.loc[0, 'payment_value'] == 35
    assert result.loc[0, 'review_score'] == 2
    assert result.loc[0, 'review_records'] == 2
    assert result.loc[0, 'category'] == 'B'


def test_first_customer_order_uses_dates_not_input_row_order():
    df = pd.DataFrame([
        order(order_id='later', order_purchase_timestamp='2018-07-01'),
        order(order_id='first', order_purchase_timestamp='2018-06-01')])
    assert first_per_customer(df).order_id.tolist() == ['first']


def test_model_splits_exclude_future_labels_and_repeated_customers():
    rows = [
        order(),
        order(order_id='o2', customer_unique_id='u2', order_purchase_timestamp='2018-06-25',
              order_approved_at='2018-06-25', order_delivered_carrier_date='2018-06-26',
              order_delivered_customer_date='2018-07-10',
              order_estimated_delivery_date='2018-07-05'),
        order(order_id='o3', customer_unique_id='u1', order_purchase_timestamp='2018-08-01',
              order_approved_at='2018-08-01', order_delivered_carrier_date='2018-08-02',
              order_delivered_customer_date='2018-08-04', order_estimated_delivery_date='2018-08-05'),
        order(order_id='o4', customer_unique_id='u4', order_purchase_timestamp='2018-08-03',
              order_approved_at='2018-08-03', order_delivered_carrier_date='2018-08-04',
              order_delivered_customer_date='2018-08-10', order_estimated_delivery_date='2018-08-12'),
    ]
    frame = engineer_features(pd.DataFrame(rows))
    train, validation, refit, test, counts = prepare_model_splits(frame)
    assert 'o2' not in train.order_id.tolist()  # target unknown on July 1
    assert 'o2' in refit.order_id.tolist()     # target known by August 1
    assert test.order_id.tolist() == ['o4']   # repeat customer u1 is not reused
    assert not set(train.customer_unique_id) & set(test.customer_unique_id)
    assert train.label_available_at.lt('2018-07-01').all()


def test_proportion_comparison_sign_and_bounds():
    result = compare_proportions(30, 100, 10, 100)
    assert result['effect'] == pytest.approx(.20)
    assert result['ci_low'] < .20 < result['ci_high']
    assert result['p_value'] < .001
    same = compare_proportions(10, 100, 10, 100)
    assert same['p_value'] == pytest.approx(1)


def test_holm_controls_the_family_and_keeps_original_order():
    np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])


def test_wilson_handles_zero_or_all_events():
    lo, hi = wilson_interval(0, 100)
    assert lo == pytest.approx(0, abs=1e-15)
    assert 0 < hi < .05
    lo, hi = wilson_interval(100, 100)
    assert .95 < lo < 1
    assert hi == pytest.approx(1)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_smaller_mde_requires_more_observations():
    assert ab_sample_size(.10, .01) > ab_sample_size(.10, .02)
