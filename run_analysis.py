"""One command: raw CSV -> audited SQL mart -> statistics -> model -> reports."""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.data import (build_mart, delivery_sample, engineer_features,
                      first_per_customer, quality_summary, write_json)
from src.model import train_and_evaluate
from src.plots import generate_figures
from src.reporting import create_report
from src.statistics import (ab_sample_size, cluster_bootstrap_review_difference,
                             diagnostics, hypothesis_tests, review_comparison,
                             standardized_difference)


def run(root=None, verbose=True):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    tables_dir = root / 'reports/tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    if verbose: print('1/5 Load and audit source tables; build order-level SQL mart')
    mart, source_audit, manifest, sql_monthly = build_mart(root)
    df = engineer_features(mart)
    delivered = delivery_sample(df)
    discovery, validation = delivery_sample(df, 'discovery'), delivery_sample(df, 'validation')
    if verbose: print('2/5 EDA and prespecified validation tests')
    hypotheses, h1_sample, h2_sample, destination_summary, destinations = hypothesis_tests(
        discovery, validation)
    diagnostic_results = diagnostics(discovery)
    sensitivity = []
    for definition, mask in [
        ('post_delivery_0_to_30_days', validation.eligible_review),
        ('all_observed_first_reviews', validation.has_review),
        ('post_delivery_any_time', validation.has_review & validation.review_lag_days.ge(0)),
    ]:
        sample = first_per_customer(validation.loc[mask])
        sensitivity.append({'definition': definition, **review_comparison(sample)})
    sensitivity = pd.DataFrame(sensitivity)
    review_coverage = (validation.groupby('is_late').agg(
        delivered_orders=('order_id', 'size'), has_review=('has_review', 'mean'),
        post_delivery_30d_review=('eligible_review', 'mean'),
        review_before_delivery=('review_lag_days', lambda s: s.lt(0).mean()))
        .reset_index())
    cluster = cluster_bootstrap_review_difference(h1_sample)
    standardized = standardized_difference(h1_sample)
    if verbose: print('3/5 Fit models with chronological splits and mature-label cutoffs')
    model_metrics, selection, model_card, predictions = train_and_evaluate(delivered)
    if verbose: print('4/5 Save analytical tables and planning scenarios')
    quality = quality_summary(df)
    monthly = (delivered.groupby('purchase_month').agg(
        orders=('order_id', 'size'), late_share=('is_late', 'mean'),
        median_delivery_days=('delivery_days', 'median'),
        mean_delivery_days=('delivery_days', 'mean')).reset_index())
    # Independent SQL/Python implementations must agree on denominators,
    # calendar-date lateness and monthly means.
    assert sql_monthly.purchase_month.tolist() == monthly.purchase_month.tolist()
    np.testing.assert_array_equal(sql_monthly.delivered_orders, monthly.orders)
    np.testing.assert_allclose(sql_monthly.late_share, monthly.late_share, atol=1e-12)
    np.testing.assert_allclose(sql_monthly.mean_delivery_days,
                               monthly.mean_delivery_days, atol=1e-8)
    ab_planning = pd.DataFrame([
        {'baseline_assumption': p, 'target_assumption': p-.02,
         'mde_absolute': .02, 'alpha': .05, 'power': .8,
         'n_per_arm': ab_sample_size(p)} for p in [.05, .10, .20]])
    summary = {
        'source_orders': len(df), 'delivered_orders': len(delivered),
        'late_share': float(delivered.is_late.mean()),
        'median_delivery_days': float(delivered.delivery_days.median()),
        'discovery_orders': len(discovery), 'validation_orders': len(validation),
        'hypotheses_plan_sha256': hashlib.sha256(
            (root / 'docs/hypotheses.md').read_bytes()).hexdigest(),
    }
    result = {
        'summary': summary, 'quality': quality, 'source_audit': source_audit,
        'monthly': monthly, 'sql_monthly': sql_monthly, 'hypotheses': hypotheses,
        'destination_summary': destination_summary, 'destinations': destinations,
        'diagnostics': diagnostic_results, 'review_sensitivity': sensitivity,
        'review_coverage': review_coverage, 'cluster_bootstrap': cluster,
        'standardized': standardized, 'model_metrics': model_metrics,
        'model_selection': selection, 'model_card': model_card,
        'ab_planning': ab_planning,
    }
    for key, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(tables_dir / f'{key}.csv', index=False)
    write_json(root / 'reports/results.json', {
        k: (v.astype(object).where(pd.notna(v), None).to_dict('records')
            if isinstance(v, pd.DataFrame) else v) for k, v in result.items()})
    write_json(root / 'reports/model_card.json', model_card)
    write_json(root / 'data/source_manifest.json', {
        'dataset': 'olistbr/brazilian-ecommerce',
        'source_url': 'https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce',
        'license': 'CC BY-NC-SA 4.0', 'files': manifest})
    write_json(root / 'reports/environment.json', {
        'python': platform.python_version(), 'pandas': pd.__version__,
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__,
        'seed': 42,
    })
    if verbose: print('5/5 Render seven figures and portable reports')
    figures = generate_figures(root, delivered, discovery, validation, h1_sample,
                                destination_summary, predictions)
    create_report(root, result)
    if verbose:
        print(hypotheses[['hypothesis', 'effect', 'p_holm']].to_string(index=False))
        print(model_metrics[['model', 'roc_auc', 'average_precision']].to_string(index=False))
        print('Ready: reports/summary.html and reports/summary.md')
    # Frames are returned for transparent exploration in the notebook, not saved
    # to the public archive as order-level data.
    result.update({'orders': df, 'delivered': delivered, 'discovery': discovery,
                   'validation': validation, 'h1_sample': h1_sample,
                   'figures': figures})
    return result


if __name__ == '__main__':
    run()
