"""Generate an evidence-linked Markdown report and a portable HTML report."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd


def number(value):
    return f'{int(value):,}'.replace(',', ' ')


def pvalue(value):
    return '< 1e-300' if value == 0 else f'{value:.3g}'


def markdown_table(df):
    # Avoid an optional tabulate dependency.
    def render(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, float):
            return f'{value:.4g}'
        return str(value).replace('|', '/')
    rows = ['| ' + ' | '.join(map(str, df.columns)) + ' |',
            '| ' + ' | '.join(['---']*len(df.columns)) + ' |']
    rows += ['| ' + ' | '.join(render(v) for v in row) + ' |'
             for row in df.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def create_report(root, result):
    summary = result['summary']
    tests = result['hypotheses'].set_index('hypothesis')
    h1, h2 = tests.loc['H1'], tests.loc['H2']
    model = result['model_metrics'].set_index('model').loc['Logistic regression']
    sensitivities = result['review_sensitivity']
    broad = sensitivities.loc[sensitivities.definition.eq('all_observed_first_reviews')].iloc[0]
    cluster = result['cluster_bootstrap']
    adjustment = result['standardized']
    planned_n = result['ab_planning'].loc[
        result['ab_planning'].baseline_assumption.eq(.10), 'n_per_arm'].iloc[0]
    report = f'''# Olist: опыт доставки и низкие оценки

Учебное продуктовое исследование открытых исторических данных Olist. Дата анализа: 28 октября 2026. Данные не принадлежат Авито; выводы не описывают пользователей Авито.

## Решение для продукта

Проверить, заслуживает ли сценарий ожидания опаздывающего заказа отдельного продуктового эксперимента. Рассматриваем прозрачный статус доставки и проактивное уведомление о риске задержки. Исследование оценивает связи и приоритет дальнейшей проверки; внедрения и измеренного прироста бизнес-метрик здесь нет.

## Результат в цифрах

- Исходные данные: **{number(summary['source_orders'])} заказов**, 8 использованных таблиц; отдельная географическая таблица не нужна.
- Основная описательная выборка: **{number(summary['delivered_orders'])}** пригодных доставленных заказов, покупки с января 2017 по август 2018; опоздания — **{summary['late_share']:.2%}**.
- H1, независимая проверочная часть: июль–август 2018, первый пригодный заказ каждого покупателя и первый отзыв в 0–30 дней после доставки. Низкие оценки: **{h1.rate_a:.2%} при опоздании против {h1.rate_b:.2%} вовремя**, разность **{h1.effect*100:+.2f} п.п.**, 95% интервал **[{h1.ci_low*100:.2f}; {h1.ci_high*100:.2f}] п.п.**, p с поправкой Холма **{pvalue(h1.p_holm)}**.
- H2: межштатная доставка в проверочном периоде в среднем длиннее на **{h2.effect:.2f} дня**, 95% интервал **[{h2.ci_low:.2f}; {h2.ci_high:.2f}]**. Это сравнение наблюдаемых маршрутов, а не эффект переноса склада.
- ML, финальный августовский тест: ROC-AUC **{model.roc_auc:.3f}**, Average Precision **{model.average_precision:.3f}** при доле опозданий **{model.late_prevalence:.2%}**. В верхних 10% оценок риска precision **{model.precision_top_10pct:.2%}**, recall **{model.recall_top_10pct:.2%}**. Это ретроспективная оценка, не результат запуска в продукте.

## Данные и определения

Источник: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), лицензия данных CC BY-NC-SA 4.0. Карточка источника объясняет: опрос может отправляться после доставки **или наступления обещанного срока**. Поэтому ответ до доставки может описывать реальный опыт ожидания.

- Гранулярность SQL-витрины: **одна строка — один заказ**. Товары и оплаты сначала агрегируются, затем соединяются с заказами. Итоги сверяются с исходными таблицами.
- Первый отзыв: самый ранний ответ покупателя по заказу; последующие ответы не добавляют независимых наблюдений.
- Опоздание: фактическая **календарная дата** доставки позже обещанной даты. Любой час обещанного дня считается своевременным.
- Низкая оценка: 1–2 из 5. Основная H1 условна на наличие первого ответа в течение 30 дней после доставки.
- Отсутствующий отзыв не заменяется положительной оценкой. Повторный customer_unique_id не считается дубликатом данных.
- В статистических выборках один заказ на покупателя. Общие продавцы всё ещё могут создавать зависимость, поэтому H1 дополнительно проверена кластерным bootstrap.

![Покрытие данных](figures/01_sample_coverage.png)

## Качество и отбор

{markdown_table(result['quality'])}

Счётчики описывают разные срезы и не все складываются друг с другом. Длинные сроки сохранены; исключены пропуски обязательных полей и невозможный порядок дат. Расхождения суммы оплат и стоимости товаров с доставкой отмечены, но не используются как автоматическая причина удаления.

## EDA: распределения и сегменты

Разведочный период: 2017-01-01 — 2018-06-30. Проверочный: 2018-07-01 — 2018-08-31. Гипотезы зафиксированы в [docs/hypotheses.md](../docs/hypotheses.md) до расчёта результатов.

![Динамика](figures/02_monthly_delivery.png)

![Распределение и Q-Q](figures/03_distribution_qq.png)

![ECDF по маршрутам](figures/04_route_ecdf.png)

Шапиро — Уилка рассчитан на воспроизводимой подвыборке из {result['diagnostics']['shapiro_n']} сроков доставки только в discovery. Он описывает несоответствие нормальной форме; не используется как автоматическое правило выбора теста. Двухвыборочный Колмогоров — Смирнов сравнивает распределения внутри/между штатами; D={result['diagnostics']['ks_d']:.3f}, разведочный p={pvalue(result['diagnostics']['ks_p_exploratory'])}. Непрерывность и совпадающие значения — ограничения калибровки KS.

## Три основные проверки

Методы: H1 — двусторонний z-тест двух долей; H2 — t-тест Уэлча; H3 — ANOVA Уэлча. Для семьи из трёх p-value применена поправка Холма при alpha=0.05. Интервалы H1/H2 — индивидуальные 95%, не одновременные интервалы для всей семьи. Подробные значения сохранены в tables/hypothesis_tests.csv. Значение p=0 в машинном CSV означает численное округление очень малого p, не нулевую вероятность.

{markdown_table(result['hypotheses'][['hypothesis', 'method', 'n_a', 'n_b', 'effect', 'p_value', 'p_holm', 'reject_h0_05']])}

![Опоздания и низкие оценки](figures/05_late_and_reviews.png)

H1: сравнение относится к ответившим после доставки. Оно **не** означает, что устранение опоздания снизит долю низких оценок ровно на эту величину.

![Три штата](figures/06_destination_means.png)

Штаты для H3 выбраны по числу заказов в discovery: {', '.join(result['destinations'])}. ANOVA показывает различия хотя бы части средних; отдельные пары после неё формально не тестировались.

## Проверки устойчивости и важный эффект отбора

{markdown_table(sensitivities[['definition', 'n_a', 'n_b', 'rate_a', 'rate_b', 'effect', 'ci_low', 'ci_high']])}

При включении **всех первых наблюдаемых отзывов**, в том числе до доставки, доли низких оценок составляют **{broad.rate_a:.2%} против {broad.rate_b:.2%}**, разность **{broad.effect*100:.2f} п.п.**. Выборки разных определений немного различаются по покупателям. Основной узкий анализ не охватывает весь отрицательный опыт ожидания: ограничение «только после доставки» отбирает отдельную группу. Направление связи устойчиво; её размер нельзя считать универсальным.

{markdown_table(result['review_coverage'])}

Кластерный bootstrap H1 на заказах с одним продавцом: {number(cluster['orders'])} заказов, {number(cluster['sellers'])} продавцов, {cluster['repeats']} повторов. Разность **{cluster['effect']*100:.2f} п.п.**, 95% percentile-интервал **[{cluster['ci_low']*100:.2f}; {cluster['ci_high']*100:.2f}] п.п.**. Это дополнительная проверка на подмножестве, не замена анализа всех заказов.

Стандартизация по месяцу и штату назначения: разность **{adjustment['effect']*100:.2f} п.п.** на общих стратах; покрытие **{adjustment['coverage']:.1%}** исходной H1. Ограниченное покрытие и ненаблюдаемые факторы остаются. Это не каузальная оценка.

## Модель как инструмент приоритизации

Задача: оценить риск опоздания в момент подтверждения оплаты. Используются только доступные к тому моменту характеристики заказа, маршрута и обещанного срока. Реальные даты доставки, отзывы и фактические сроки запрещены в признаках.

Обучение — до июля 2018, выбор C из 0.1/1/10 по июльскому Average Precision, переобучение на доступных к 1 августа метках, финальный тест — август. Покупатель входит только в один временной split. Для обучения и июльской валидации исключены метки, неизвестные на соответствующую дату отсечения; это создаёт отбор по зрелости меток, явно отражённый в model_card.json.

{markdown_table(result['model_metrics'])}

![Precision-recall](figures/07_model_precision_recall.png)

Отменённые и недоставленные заказы в ML-выборку не входят. Поэтому модель относится к заказам, которые впоследствии были доставлены; внедрение потребовало бы отдельного определения исходов для всех подтверждённых заказов. На августе Brier score логистической регрессии хуже константной базы, хотя ROC-AUC и Average Precision выше: ранжирование полезнее, но качество вероятностей требует отдельной калибровки и проверки. Реальные затраты обработки не проверены. Метрики top-10% — сценарий ограниченной мощности, не готовый онлайн-порог.

## Приоритет и следующий продуктовый шаг

**Приоритет: исследовать прозрачность статуса и проактивные уведомления при риске задержки.** Основания: устойчивая связь задержек с низкими оценками, наличие отзывов ещё до доставки и возможность сформировать риск-сегмент. Уведомление не сокращает физический срок доставки автоматически.

| Кандидат | Основание | Что неизвестно | Решение |
| --- | --- | --- | --- |
| Проактивный статус при риске задержки | H1 и отзывы до доставки | Снизятся ли обращения; не вырастет ли тревожность | Первый кандидат на исследование и A/B |
| Изменение обещанного срока | Различия маршрутов/штатов | Влияние на конверсию и доверие | Нужны события просмотра и оформления |
| Перестройка логистики | Межштатные доставки длиннее | Затраты, ресурсы, причинный эффект | Передать в отдельное операционное исследование |

Приоритизация качественная: данные не содержат затрат и ожидаемого causal uplift, поэтому баллы ICE/RICE не выдумываются.

## План A/B-теста: проектирование, не проведённый эксперимент

- Изменение: понятный статус и проактивное уведомление для заранее определённого риск-сегмента.
- Единица рандомизации: покупатель; группа закрепляется за customer_unique_id. В основном анализе один первый подходящий заказ на покупателя.
- Основная метрика: доля покупателей с обращением в поддержку по доставке до обещанной даты + 30 календарных дней; знаменатель — все рандомизированные покупатели (ITT).
- Защитные метрики: отмены, доля успешно доставленных заказов, отписки от уведомлений, стоимость поддержки на заказ. Оценки и доля ответов на опрос — дополнительные метрики; отдельно контролируем пропуски и изменение отклика.
- Новое логирование: assignment, eligibility/risk snapshot, notification sent/delivered, support contact with reason, cancellation, delivery, survey offered/answered. В открытых данных Olist **нет обращений в поддержку**, поэтому эффект и baseline основной экспериментальной метрики не измерены.
- Плановый alpha=0.05, power=0.80, двусторонняя проверка, MDE=2 п.п. Эти параметры — явные допущения для расчёта, не требования бизнеса.
- При **условном baseline 10%** и цели 8% требуется примерно **{number(planned_n)} покупателей на группу**. Это пример планирования; перед запуском baseline надо измерить в продукте, а MDE согласовать с командой. Не переводим размер выборки в дни без данных о трафике риск-сегмента.
- До оценки эффекта проверить баланс/SRM, полноту логов и одинаковые окна наблюдения. Продолжительность фиксируется заранее с учётом набора выборки, недельного цикла и 30-дневного окна; не останавливать тест при первом p < 0.05.

{markdown_table(result['ab_planning'])}

## Практические ограничения

Наблюдательные данные одного исторического маркетплейса; нет случайного назначения задержек. Связи могут объясняться продавцами, категорией, географией, сезоном и другими факторами. Нет полного трафика, событий корзины, стоимости привлечения, расходов на поддержку и данных всех онлайн-экспериментов; CR сайта, причинный retention uplift и прибыль здесь не рассчитываются. Низкие оценки характеризуют только наблюдаемые ответы. Общие продавцы и временные факторы создают зависимость; выводы H2/H3 условны на предпосылки тестов. Для Авито отдельно проверяется переносимость постановки и метрик.

## Источники и воспроизводимость

- [Olist, карточка и схема данных](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- [SciPy: Shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html), [KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html), [Welch t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html), [ANOVA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html).
- [Newcombe 1998: интервалы разности независимых долей](https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C873::AID-SIM779%3E3.0.CO;2-I).
- [scikit-learn: метрики классификации](https://scikit-learn.org/stable/modules/model_evaluation.html).
- Сводные таблицы: reports/tables; контрольные суммы CSV: data/source_manifest.json; правила и план: docs/hypotheses.md.
'''
    (root / 'reports/summary.md').write_text(report, encoding='utf-8')
    # Self-contained HTML: no CDN, network requests, scripts or external fonts.
    rendered = []
    in_list, in_table = False, False
    lines = report.splitlines()
    for line in lines:
        if line.startswith('| '):
            if not in_table:
                rendered.append('<div class="tablewrap"><table>')
                in_table = True
            if set(line.replace('|', '').replace(' ', '')) <= {'-', ':'}:
                continue
            cells = [html.escape(c.strip()) for c in line.strip('|').split('|')]
            rendered.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            continue
        if in_table:
            rendered.append('</table></div>'); in_table = False
        if line.startswith('- '):
            if not in_list:
                rendered.append('<ul>'); in_list = True
            rendered.append('<li>' + inline(line[2:]) + '</li>'); continue
        if in_list:
            rendered.append('</ul>'); in_list = False
        if line.startswith('!['):
            name = line.split('](', 1)[1].rstrip(')')
            image_path = root / 'reports' / name
            encoded = base64.b64encode(image_path.read_bytes()).decode()
            rendered.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="График анализа"></figure>')
        elif line.startswith('# '):
            rendered.append('<h1>' + inline(line[2:]) + '</h1>')
        elif line.startswith('## '):
            rendered.append('<h2>' + inline(line[3:]) + '</h2>')
        elif line.strip():
            rendered.append('<p>' + inline(line) + '</p>')
    if in_list: rendered.append('</ul>')
    if in_table: rendered.append('</table></div>')
    css = '''body{margin:0;background:#edf2f4;color:#182e3a;font:16px/1.65 system-ui,sans-serif}
main{max-width:1000px;margin:32px auto;background:#fff;padding:44px 54px;border-radius:16px}
.eyebrow{color:#087f8c;letter-spacing:.14em;font-size:12px;font-weight:700}
h1{font-size:38px;line-height:1.2;max-width:800px;margin:14px 0 28px}h2{font-size:24px;line-height:1.3;margin:40px 0 16px;border-top:1px solid #dce4e8;padding-top:24px}
a{color:#087f8c}strong{color:#123c49}li{margin:8px 0}img{width:100%;height:auto}figure{margin:24px 0}
.tablewrap{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:13px}td{padding:10px 12px;border-bottom:1px solid #dce4e8;vertical-align:top}tr:first-child{background:#e9f4f5;font-weight:700}tr:nth-child(even){background:#f8fafb}
@media(max-width:700px){main{margin:0;padding:24px 18px;border-radius:0}h1{font-size:29px}h2{font-size:22px}body{font-size:15px}}
@media print{body{background:white}main{margin:0;padding:0;max-width:none}h2{break-after:avoid}figure{break-inside:avoid}}'''
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Olist — опыт доставки и низкие оценки</title>'
            f'<style>{css}</style><main><div class="eyebrow">PRODUCT ANALYTICS · OLIST · 2026</div>'
            + '\n'.join(rendered) + '</main></html>')
    (root / 'reports/summary.html').write_text(page, encoding='utf-8')
    return report


def inline(text):
    import re
    escaped = html.escape(text)
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', escaped)
    return escaped
