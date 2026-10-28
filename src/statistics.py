"""Effect sizes, uncertainty and prespecified hypothesis checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import first_per_customer


def wilson_interval(successes, total, alpha=0.05):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError('Require 0 <= successes <= total and total > 0')
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z*z / total
    middle = (p + z*z / (2*total)) / denominator
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0.0, middle-radius), min(1.0, middle+radius)


def compare_proportions(k_a, n_a, k_b, n_b):
    """A - B; pooled two-sided z p-value, unpooled Newcombe-Wilson CI."""
    a_low, a_high = wilson_interval(k_a, n_a)
    b_low, b_high = wilson_interval(k_b, n_b)
    p_a, p_b = k_a/n_a, k_b/n_b
    pooled = (k_a+k_b)/(n_a+n_b)
    se = np.sqrt(pooled*(1-pooled)*(1/n_a+1/n_b))
    z = (p_a-p_b)/se if se > 0 else 0.0
    difference = p_a-p_b
    return {
        'n_a': int(n_a), 'n_b': int(n_b),
        'rate_a': p_a, 'rate_b': p_b, 'effect': difference,
        'ci_low': max(-1.0, difference - np.hypot(p_a-a_low, b_high-p_b)),
        'ci_high': min(1.0, difference + np.hypot(a_high-p_a, p_b-b_low)),
        'statistic': float(z), 'p_value': float(2*stats.norm.sf(abs(z))),
    }


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('p-values must be finite values between 0 and 1')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def review_comparison(df):
    late = df.loc[df.is_late, 'low_review']
    on_time = df.loc[~df.is_late, 'low_review']
    return compare_proportions(int(late.sum()), len(late),
                               int(on_time.sum()), len(on_time))


def hypothesis_tests(discovery, validation):
    # Choose the three destinations using discovery counts only.
    destinations = (first_per_customer(discovery).customer_state.value_counts()
                    .head(3).index.tolist())
    h1_sample = first_per_customer(validation.loc[validation.eligible_review])
    h1 = {'hypothesis': 'H1', 'method': 'two_proportion_z',
          'effect_unit': 'share_difference_late_minus_on_time',
          **review_comparison(h1_sample)}
    h2_sample = first_per_customer(validation.loc[
        validation.route.isin(['same_state', 'cross_state'])])
    cross = h2_sample.loc[h2_sample.route.eq('cross_state'), 'delivery_days']
    same = h2_sample.loc[h2_sample.route.eq('same_state'), 'delivery_days']
    welch = stats.ttest_ind(cross, same, equal_var=False)
    ci = welch.confidence_interval()
    h2 = {'hypothesis': 'H2', 'method': 'welch_t',
          'effect_unit': 'days_cross_minus_same',
          'n_a': len(cross), 'n_b': len(same),
          'rate_a': float(cross.mean()), 'rate_b': float(same.mean()),
          'effect': float(cross.mean()-same.mean()),
          'ci_low': float(ci.low), 'ci_high': float(ci.high),
          'statistic': float(welch.statistic), 'p_value': float(welch.pvalue)}
    h3_sample = first_per_customer(validation.loc[
        validation.customer_state.isin(destinations)])
    groups = [h3_sample.loc[h3_sample.customer_state.eq(s), 'delivery_days']
              for s in destinations]
    anova = stats.f_oneway(*groups, equal_var=False)
    group_summary = (h3_sample.groupby('customer_state').delivery_days
                     .agg(['size', 'mean', 'median', 'std']).reset_index())
    # A raw range of group means is descriptive, not an ANOVA effect estimate.
    h3 = {'hypothesis': 'H3', 'method': 'welch_anova',
          'effect_unit': 'see_destination_means_table',
          'n_a': len(h3_sample), 'n_b': 0,
          'rate_a': None, 'rate_b': None, 'effect': None,
          'ci_low': None, 'ci_high': None,
          'statistic': float(anova.statistic), 'p_value': float(anova.pvalue)}
    result = pd.DataFrame([h1, h2, h3])
    result['p_holm'] = holm_adjust(result.p_value)
    result['reject_h0_05'] = result.p_holm.lt(0.05)
    return result, h1_sample, h2_sample, group_summary, destinations


def diagnostics(discovery, seed=42):
    # First order/customer; diagnostics stay entirely inside discovery.
    sample = first_per_customer(discovery)
    durations = sample.delivery_days.to_numpy()
    rng = np.random.default_rng(seed)
    shapiro_sample = rng.choice(durations, min(5000, len(durations)), replace=False)
    shapiro = stats.shapiro(shapiro_sample)
    cross = sample.loc[sample.route.eq('cross_state'), 'delivery_days']
    same = sample.loc[sample.route.eq('same_state'), 'delivery_days']
    ks = stats.ks_2samp(cross, same, method='asymp')
    return {
        'shapiro_n': len(shapiro_sample), 'shapiro_w': float(shapiro.statistic),
        'shapiro_p': float(shapiro.pvalue),
        'ks_d': float(ks.statistic), 'ks_p_exploratory': float(ks.pvalue),
        'ks_cross_n': len(cross), 'ks_same_n': len(same),
        'interpretation': 'Discovery diagnostics; not a test-selection gate. '
                          'KS assumes continuous distributions; ties can affect calibration.',
    }


def cluster_bootstrap_review_difference(sample, repeats=2000, seed=42):
    # Single-seller restriction makes each order belong to one clear cluster.
    df = sample.loc[sample.seller_count.eq(1)].copy()
    df['late_low'] = df.is_late & df.low_review
    df['ontime'] = ~df.is_late
    df['ontime_low'] = df.ontime & df.low_review
    totals = df.groupby('main_seller_id')[
        ['is_late', 'late_low', 'ontime', 'ontime_low']].sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sums = totals[rng.integers(0, len(totals), size=len(totals))].sum(axis=0)
        if sums[0] and sums[2]:
            estimates.append(sums[1]/sums[0]-sums[3]/sums[2])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {'method': 'seller_cluster_percentile_bootstrap', 'orders': len(df),
            'sellers': len(totals), 'repeats': repeats, 'seed': seed,
            'effect': review_comparison(df)['effect'],
            'ci_low': float(low), 'ci_high': float(high)}


def standardized_difference(sample, minimum_per_group=10):
    # Weights are the combined order mix in common month/state strata.
    a = (sample.groupby(['purchase_month', 'customer_state', 'is_late'])
         .low_review.agg(['sum', 'size']).unstack('is_late').fillna(0))
    common = a.loc[(a[('size', True)] >= minimum_per_group)
                   & (a[('size', False)] >= minimum_per_group)]
    weights = common['size'].sum(axis=1)
    effects = (common[('sum', True)]/common[('size', True)]
               - common[('sum', False)]/common[('size', False)])
    covered = int(weights.sum())
    return {'strata': len(common), 'covered_orders': covered,
            'coverage': covered/len(sample),
            'effect': float(np.average(effects, weights=weights)),
            'method': 'standardization_month_and_customer_state',
            'limitation': 'Observed mix adjustment only; unmeasured confounding remains.'}


def ab_sample_size(baseline, mde_absolute=0.02, alpha=0.05, power=0.8):
    """Normal approximation, two-sided test of two independent proportions."""
    if not 0 < baseline-mde_absolute < baseline < 1:
        raise ValueError('Require 0 < baseline - MDE < baseline < 1')
    p1, p2 = baseline, baseline-mde_absolute
    pbar = (p1+p2)/2
    z_alpha, z_power = stats.norm.ppf(1-alpha/2), stats.norm.ppf(power)
    n = ((z_alpha*np.sqrt(2*pbar*(1-pbar))
          + z_power*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2 / mde_absolute**2)
    return int(np.ceil(n))
