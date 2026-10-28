"""Static, exportable figures with explicit denominators and periods."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_curve

from .statistics import wilson_interval

TEAL, ORANGE, INK, GREY = '#087f8c', '#e88343', '#182e3a', '#6e7e87'


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'figure.facecolor': '#fbfcfd', 'axes.facecolor': '#fbfcfd',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#ccd6da', 'text.color': INK,
        'axes.labelcolor': INK, 'xtick.color': GREY, 'ytick.color': GREY,
        'axes.titleweight': 'bold', 'axes.titlesize': 13,
        'grid.color': '#dce4e8', 'grid.alpha': 0.6,
        'savefig.dpi': 150, 'figure.constrained_layout.use': True,
    })


def save(fig, directory, name):
    path = directory / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def generate_figures(root, delivered, discovery, validation, h1_sample,
                     destination_summary, predictions):
    style()
    out = root / 'reports/figures'
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(9, 4.5))
    counts = [len(delivered), int(delivered.has_review.sum()),
              int(delivered.eligible_review.sum()), len(h1_sample)]
    labels = ['Пригодные доставленные заказы\n2017-01 — 2018-08',
              'Есть первый отзыв\nв основном периоде',
              'Первый отзыв в 0–30 дней\nпосле доставки',
              'Выборка H1: июль–август,\nодин заказ на покупателя']
    bars = ax.barh(labels[::-1], counts[::-1], color=[ORANGE, TEAL, TEAL, TEAL])
    ax.bar_label(bars, labels=[f'{x:,}'.replace(',', ' ') for x in counts[::-1]], padding=6)
    ax.set_xlim(0, max(counts)*1.15)
    ax.set_title('Покрытие данных и выборка основной проверки', loc='left')
    ax.set_xlabel('Число заказов; это отбор наблюдений, не продуктовая воронка')
    ax.grid(axis='x'); ax.set_axisbelow(True)
    paths.append(save(fig, out, '01_sample_coverage.png'))

    monthly = delivered.groupby('purchase_month').agg(
        orders=('order_id', 'size'), late_share=('is_late', 'mean'))
    x = np.arange(len(monthly))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(x, monthly.orders, color=TEAL, alpha=.85)
    axes[0].set_ylabel('Заказы')
    axes[0].set_title('Объём и доля опозданий по месяцу покупки', loc='left')
    axes[1].plot(x, monthly.late_share*100, 'o-', color=ORANGE)
    axes[1].set_ylabel('Опоздания, %')
    for ax in axes:
        ax.axvspan(len(x)-2.5, len(x)-.5, color=ORANGE, alpha=.10)
        ax.grid(axis='y'); ax.set_axisbelow(True)
    axes[1].set_xticks(x[::2], monthly.index[::2], rotation=35, ha='right')
    axes[1].set_xlabel('Затенение: проверочный период июль–август 2018')
    paths.append(save(fig, out, '02_monthly_delivery.png'))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    durations = discovery.delivery_days
    axes[0].hist(durations[durations.le(80)], bins=55, color=TEAL, alpha=.9)
    axes[0].set_title('Распределение сроков доставки', loc='left')
    axes[0].set_xlabel('Дни от покупки до доставки'); axes[0].set_ylabel('Заказы')
    stats.probplot(durations, dist='norm', plot=axes[1])
    axes[1].get_lines()[0].set(color=TEAL, markersize=2, alpha=.5)
    axes[1].get_lines()[1].set(color=ORANGE)
    axes[1].set_title('')  # probplot sets a centred English title; remove it.
    axes[1].set_title('Q–Q нормального распределения', loc='left', fontsize=12)
    axes[1].set_xlabel('Теоретические квантили'); axes[1].set_ylabel('Срок доставки, дни')
    fig.suptitle('Discovery: длинный хвост сохранён в расчётах; гистограмма до 80 дней', fontsize=11)
    paths.append(save(fig, out, '03_distribution_qq.png'))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for route, label, color in [('same_state', 'Внутри штата', TEAL),
                                ('cross_state', 'Между штатами', ORANGE)]:
        values = np.sort(discovery.loc[discovery.route.eq(route), 'delivery_days'])
        ax.plot(values, np.arange(1, len(values)+1)/len(values),
                label=f'{label} (n={len(values):,})', color=color)
    ax.set_xlim(0, 60); ax.set_ylim(0, 1)
    ax.set_xlabel('Срок доставки, дни'); ax.set_ylabel('Накопленная доля заказов')
    ax.set_title('Форма распределений по маршруту — discovery', loc='left')
    ax.legend(loc='lower right'); ax.grid()
    paths.append(save(fig, out, '04_route_ecdf.png'))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (late, label, color) in enumerate([
        (False, 'Вовремя', TEAL), (True, 'С опозданием', ORANGE)]):
        group = h1_sample.loc[h1_sample.is_late.eq(late)]
        rate = group.low_review.mean()
        lo, hi = wilson_interval(int(group.low_review.sum()), len(group))
        ax.bar(i, rate*100, width=.55, color=color)
        ax.errorbar(i, rate*100, yerr=[[100*(rate-lo)], [100*(hi-rate)]],
                    color=INK, capsize=6, fmt='none')
        ax.text(i, hi*100+2, f'{rate:.1%}\nn={len(group):,}', ha='center', fontsize=11)
    ax.set_xticks([0, 1], ['Вовремя', 'С опозданием'])
    ax.set_ylim(0, 30); ax.set_ylabel('Оценки 1–2 среди пригодных отзывов, %')
    ax.set_title('H1: низкие оценки и опоздание — июль–август 2018', loc='left')
    ax.set_xlabel('Один заказ на покупателя; интервалы Уилсона 95%; связь, не причинный эффект')
    ax.grid(axis='y'); ax.set_axisbelow(True)
    paths.append(save(fig, out, '05_late_and_reviews.png'))

    summary = destination_summary.sort_values('mean')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    error = 1.96*summary['std']/np.sqrt(summary['size'])
    ax.errorbar(summary['mean'], summary.customer_state, xerr=error,
                fmt='o', markersize=8, capsize=5, color=TEAL)
    for row in summary.itertuples():
        ax.annotate(f'  {row.mean:.1f} дня; n={row.size:,}',
                    (row.mean, row.customer_state), xytext=(15, 9), textcoords='offset points')
    ax.set_xlim(0, summary['mean'].max()+6)
    ax.margins(y=.25)
    ax.set_xlabel('Средний срок доставки, дни; приближённый 95% интервал')
    ax.set_title('H3: направления выбраны по объёму discovery', loc='left')
    ax.grid(axis='x')
    paths.append(save(fig, out, '06_destination_means.png'))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    precision, recall, _ = precision_recall_curve(predictions.y_true, predictions.score)
    ax.plot(recall, precision, color=TEAL, label='Логистическая регрессия')
    ax.axhline(predictions.y_true.mean(), color=ORANGE, linestyle='--',
               label=f'База: {predictions.y_true.mean():.1%}')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall', ylabel='Precision')
    ax.set_title('Риск опоздания: финальный тест на августе 2018', loc='left')
    ax.legend(); ax.grid()
    paths.append(save(fig, out, '07_model_precision_recall.png'))
    return paths
