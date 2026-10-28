"""Predict late delivery using only features available at payment approval."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC = ['goods_value', 'freight_value', 'item_count', 'seller_count',
           'promised_days_at_approval', 'freight_share']
CATEGORICAL = ['customer_state', 'seller_state', 'category', 'purchase_weekday']
FEATURES = NUMERIC + CATEGORICAL


def prepare_model_splits(delivered):
    eligible = delivered.loc[
        delivered.order_approved_at.notna()
        & delivered.promised_days_at_approval.ge(0)
        & delivered.order_approved_at.ge(delivered.order_purchase_timestamp)
        & delivered.order_approved_at.ge('2017-01-01')
        & delivered.order_approved_at.lt('2018-09-01')].copy()
    # An order's label becomes observable by delivery, or at the start of the
    # day after the promised date if it is still undelivered. No future label
    # may be used in an as-of training or validation set.
    eligible['label_available_at'] = eligible.order_delivered_customer_date.where(
        ~eligible.is_late,
        eligible.order_estimated_delivery_date.dt.normalize() + pd.Timedelta(days=1))
    # One customer appears in only one split. Sort by scoring time.
    eligible = (eligible.sort_values(['order_approved_at', 'order_id'])
                .drop_duplicates('customer_unique_id', keep='first'))
    approved = eligible.order_approved_at
    train = eligible.loc[approved.lt('2018-07-01')
                          & eligible.label_available_at.lt('2018-07-01')].copy()
    validation = eligible.loc[approved.ge('2018-07-01') & approved.lt('2018-08-01')
                               & eligible.label_available_at.lt('2018-08-01')].copy()
    refit = eligible.loc[approved.lt('2018-08-01')
                          & eligible.label_available_at.lt('2018-08-01')].copy()
    test = eligible.loc[approved.ge('2018-08-01') & approved.lt('2018-09-01')].copy()
    counts = {
        'train': len(train), 'july_validation_mature_labels': len(validation),
        'refit_as_of_august_1': len(refit), 'august_test': len(test),
        'train_labels_unavailable_at_cutoff': int((approved.lt('2018-07-01')
            & eligible.label_available_at.ge('2018-07-01')).sum()),
        'july_labels_unavailable_at_cutoff': int((approved.ge('2018-07-01')
            & approved.lt('2018-08-01')
            & eligible.label_available_at.ge('2018-08-01')).sum()),
    }
    return train, validation, refit, test, counts


def make_pipeline(c=1.0):
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50)),
    ])
    preprocess = ColumnTransformer([
        ('numeric', numeric, NUMERIC), ('categorical', categorical, CATEGORICAL)])
    return Pipeline([
        ('preprocess', preprocess),
        ('classifier', LogisticRegression(C=c, max_iter=2000, random_state=42)),
    ])


def evaluate(y, score, seed=42):
    y, score = np.asarray(y, int), np.asarray(score, float)
    budget = int(np.ceil(len(y)*0.10))
    # A random reproducible tie-break avoids selecting the first date block
    # for a constant-score DummyClassifier.
    order = np.lexsort((np.random.default_rng(seed).random(len(y)), -score))
    selected = y[order[:budget]]
    return {
        'n': len(y), 'late_prevalence': float(y.mean()),
        'roc_auc': float(roc_auc_score(y, score)),
        'average_precision': float(average_precision_score(y, score)),
        'brier_score': float(brier_score_loss(y, score)),
        'precision_top_10pct': float(selected.mean()),
        'recall_top_10pct': float(selected.sum()/y.sum()),
        'selected_orders': budget,
    }


def train_and_evaluate(delivered):
    train, validation, refit, test, counts = prepare_model_splits(delivered)
    # Candidate grid fixed before model results; July AP is the selection metric.
    candidates = []
    for c in [0.1, 1.0, 10.0]:
        estimator = make_pipeline(c)
        estimator.fit(train[FEATURES], train.is_late.astype(int))
        score = estimator.predict_proba(validation[FEATURES])[:, 1]
        candidates.append({'C': c, **evaluate(validation.is_late, score)})
    selection = pd.DataFrame(candidates)
    chosen_c = float(selection.sort_values('average_precision', ascending=False).iloc[0].C)
    final = make_pipeline(chosen_c)
    final.fit(refit[FEATURES], refit.is_late.astype(int))
    baseline = DummyClassifier(strategy='prior')
    baseline.fit(refit[FEATURES], refit.is_late.astype(int))
    final_score = final.predict_proba(test[FEATURES])[:, 1]
    baseline_score = baseline.predict_proba(test[FEATURES])[:, 1]
    metrics = pd.DataFrame([
        {'model': 'Dummy prior', **evaluate(test.is_late, baseline_score)},
        {'model': 'Logistic regression', **evaluate(test.is_late, final_score)},
    ])
    card = {
        'task': 'Risk of late delivery at payment approval, conditional on eventual delivery',
        'chosen_C_by_july_AP': chosen_c, 'features': FEATURES,
        'split_counts': counts,
        'forbidden_features': ['order_delivered_customer_date', 'review_score',
            'delivery_days', 'late_days', 'is_late', 'review_answer_timestamp'],
        'limitations': [
            'Retrospective sample includes only orders eventually delivered; '
            'cancelled/undelivered orders require a different outcome definition.',
            'Training and July selection exclude labels unknown at the cutoff; '
            'July validation is therefore a maturity-selected subset.',
            'August is a single out-of-time test; no causal or online performance claim.',
            'Probabilities are not calibrated on a separate sample. '
            'Precision/recall at 10% describe a retrospective capacity scenario, not an online threshold.',
        ],
    }
    predictions = pd.DataFrame({'order_id': test.order_id.to_numpy(),
                                'y_true': test.is_late.astype(int).to_numpy(),
                                'score': final_score})
    return metrics, selection, card, predictions
