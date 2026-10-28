"""Build the readable notebook and optionally execute every cell from scratch."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md('''# Olist: опыт доставки и низкие оценки

**Продуктовый вопрос:** стоит ли команде проверять прозрачный статус и проактивные уведомления о риске задержки?

Используем реальные обезличенные заказы Olist, не данные Авито. Отделяем наблюдаемые связи, прогноз и будущий эксперимент.

Маршрут: контекст → качество данных → EDA → гипотезы → статистика → устойчивость → приоритет → модель → дизайн A/B.

Все графики и итоговые таблицы уже содержатся в выполненном ноутбуке. Для повторного запуска скачайте данные командой `python scripts/download_data.py`.'''),
        code('''from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = Path.cwd()
if not (ROOT / 'run_analysis.py').exists():
    ROOT = ROOT.parent
if not (ROOT / 'run_analysis.py').exists():
    raise RuntimeError('Откройте ноутбук из корня проекта или папки notebooks')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pd.set_option('display.max_columns', 14)
pd.set_option('display.float_format', lambda x: f'{x:.5g}')

# Один запуск общего пайплайна. Далее работаем с возвращёнными таблицами.
# Реализация шагов открыта в src/, SQL — в sql/.
from run_analysis import run
r = run(ROOT, verbose=True)
orders = r['orders']
delivered = r['delivered']
discovery = r['discovery']
validation = r['validation']'''),
        md('''## 1. Единица анализа и происхождение данных

Источник: [Olist на Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Лицензия данных: CC BY-NC-SA 4.0. Контрольные суммы — `data/source_manifest.json`.

Витрина имеет одну строку на заказ. `customer_id` относится к заказу; для повторных покупок нужен `customer_unique_id`. Товары/оплаты агрегируются до соединения. Из нескольких отзывов выбирается первый ответ. Географическая таблица не используется: для этой постановки достаточно штатов.

Основные поля: `delivery_days` — срок от покупки; `is_late` — доставлено после обещанной календарной даты; `low_review` — оценка 1–2; `eligible_review` — первый ответ в 0–30 дней после доставки; `route` — внутри/между штатами только для одного продавца.'''),
        code("""display(r['source_audit'])
display(orders[['order_id', 'item_count', 'seller_count', 'goods_value',
                'freight_value', 'customer_state', 'seller_state',
                'delivery_days', 'is_late', 'review_score']].head())
assert orders.order_id.is_unique
print(f'Заказов в SQL-витрине: {len(orders):,}')"""),
        md('''## 2. Проверка качества и очистка

Не удаляем длинный хвост сроков автоматически. Выделяем невозможный порядок дат и пропуски обязательных полей. Повторные покупки не являются дублями. Отзывы до доставки храним: опрос может приходить при наступлении обещанного срока.

Основная H1 отвечает узкому вопросу об ответах после получения товара. Важное ограничение этого выбора проверим отдельно.'''),
        code("""display(r['quality'])
missing = (orders.isna().mean().mul(100).sort_values(ascending=False)
           .rename('missing_pct').to_frame())
display(missing.loc[missing.missing_pct.gt(0)].head(15))
display(orders.groupby('cohort').agg(
    orders=('order_id', 'size'), valid_delivery=('valid_delivery', 'sum')))
display(Image(filename=str(ROOT/'reports/figures/01_sample_coverage.png')))"""),
        md('''## 3. Описательные метрики

Знаменатель доли опозданий — пригодные доставленные заказы. Знаменатель низких оценок — явно определённые наблюдаемые ответы. Нет событий трафика и корзины, поэтому CR сайта здесь посчитать нельзя.

Для графиков EDA используем все пригодные заказы. Для тестов дальше берём один заказ покупателя, чтобы повторные покупки не создавали ложное увеличение размера выборки.'''),
        code("""metrics = pd.Series({
    'Доставленных заказов': len(delivered),
    'Доля опозданий': delivered.is_late.mean(),
    'Медиана срока, дни': delivered.delivery_days.median(),
    'Средний срок, дни': delivered.delivery_days.mean(),
    'Покрытие отзывами 0–30 дней после доставки': delivered.eligible_review.mean(),
})
display(metrics.to_frame('value'))
display(r['monthly'])
display(Image(filename=str(ROOT/'reports/figures/02_monthly_delivery.png')))"""),
        md('''## 4. Распределение сроков, Шапиро и Колмогоров — Смирнов

Discovery: январь 2017 — июнь 2018. На гистограмме для читаемости показаны сроки до 80 дней, в расчётах длинные сроки сохранены.

Шапиро — Уилка: проверка нормальной формы на воспроизводимой подвыборке до 5000 наблюдений. Q–Q помогает увидеть характер отклонения. Малый p не означает, что любой анализ средних запрещён; при выборе метода учитываем вопрос, размеры групп и зависимость.

Двухвыборочный KS сравнивает форму распределений по маршрутам. Он используется как дополнительная диагностика в discovery. Базовая калибровка предполагает непрерывные распределения; совпадающие значения могут влиять на p.'''),
        code("""display(Image(filename=str(ROOT/'reports/figures/03_distribution_qq.png')))
display(Image(filename=str(ROOT/'reports/figures/04_route_ecdf.png')))
display(pd.Series(r['diagnostics']).to_frame('value'))"""),
        md('''## 5. Фиксируем вопросы до проверочного периода

Полный план находится в `docs/hypotheses.md`; он записан до первого расчёта результатов, это не внешняя формальная пререгистрация.

| Гипотеза | Выборка | Метод |
| --- | --- | --- |
| H1: доля оценок 1–2 различается у вовремя/опоздавших | Первый пригодный заказ/покупателя, ответ после доставки за 30 дней | z-тест двух долей; интервал разности Ньюкомба |
| H2: средний срок различается внутри/между штатами | Первый пригодный заказ/покупателя, один продавец | t-тест Уэлча |
| H3: средний срок различается между тремя массовыми штатами | Штаты выбраны только по численности discovery | ANOVA Уэлча |

Все три основные проверки — на покупках июля–августа 2018; p-value корректируем методом Холма. Выборка H1 может отличаться от H2/H3 из-за доступности отзывов.'''),
        code("""display(r['hypotheses'])
print('Штаты для H3, зафиксированные по discovery:', r['destinations'])
# Поправка Холма относится к p-value. Показанные интервалы — индивидуальные 95%.
# 0.0 для очень малого p в CSV — численное округление, не точный ноль."""),
        md('''## 6. H1: опоздание и низкие оценки

z-тест оценивает совместимость данных с равенством долей. Для интервала разности используем Ньюкомба на основе интервалов Уилсона.

Размер связи и интервал важнее одного p. Отсутствие случайного назначения задержек не позволяет трактовать различие как эффект устранения опоздания.'''),
        code("""from src.statistics import compare_proportions
h1 = r['h1_sample']
late = h1.loc[h1.is_late, 'low_review']
on_time = h1.loc[~h1.is_late, 'low_review']
display(pd.Series(compare_proportions(
    int(late.sum()), len(late), int(on_time.sum()), len(on_time))).to_frame('H1'))
display(Image(filename=str(ROOT/'reports/figures/05_late_and_reviews.png')))"""),
        md('''## 7. H2/H3: сроки по маршрутам и направлениям

Тест Уэлча сравнивает средние без предположения об одинаковых дисперсиях. ANOVA Уэлча обобщает сравнение средних на несколько групп. Она не говорит, какие пары различаются: отдельных post-hoc проверок здесь нет.

Это наблюдаемые маршруты и штаты, различающиеся ещё и по продавцам/товарам. Общие продавцы могут создавать зависимость даже после выбора одного заказа на покупателя.'''),
        code("""display(r['hypotheses'].query("hypothesis in ['H2', 'H3']"))
display(r['destination_summary'])
display(Image(filename=str(ROOT/'reports/figures/06_destination_means.png')))"""),
        md('''## 8. Устойчивость: определение отзыва заметно меняет результат

Сравниваем все первые ответы и ответы только после доставки. В группе опозданий часть отрицательного опыта возникает ещё до получения товара. Поэтому «исключить отзывы до доставки» не является универсальной очисткой.

Дополнительно проверяем H1 bootstrap по продавцам и стандартизацией по месяцу/штату. Последняя использует только общие достаточно наполненные страты; покрытие обязательно показывается.'''),
        code("""display(r['review_sensitivity'][[
    'definition', 'n_a', 'n_b', 'rate_a', 'rate_b', 'effect', 'ci_low', 'ci_high']])
display(r['review_coverage'])
display(pd.Series(r['cluster_bootstrap']).to_frame('seller_cluster_bootstrap'))
display(pd.Series(r['standardized']).to_frame('mix_adjustment'))"""),
        md('''## 9. Приоритет продуктовой гипотезы

**Кандидат:** понятный статус и проактивное уведомление для заказов с риском задержки.

Почему: связь опоздания с низкой оценкой сохраняется при разных определениях, а отзывы до доставки указывают на значимость самого ожидания. Проверка эффекта уведомлений ещё впереди.

Другие кандидаты — изменение обещанного срока и перестройка логистики. Для них не хватает событий оформления/конверсии и затрат. Не назначаем вымышленные ICE/RICE-баллы и не обещаем измеренный uplift.

Следующая ML-задача помогает выделять риск-сегмент для дальнейшей работы. Она прогнозирует опоздание; она не доказывает полезность уведомления.'''),
        md('''## 10. Дополнительная модель: риск опоздания в момент оплаты

Доступные признаки: состав и стоимость заказа, доставка, категория, штаты, обещанный оставшийся срок и день недели. Запрещены реальные даты доставки, отзывы, фактическая длительность и target.

Временные выборки: обучение до июля, выбор C по июлю, тест на августе. Метки для обучения/выбора должны быть уже известны на соответствующую дату отсечения. Покупатель не повторяется между split.

Базовая модель — константная доля, основная — логистическая регрессия. Сравниваем ROC-AUC, Average Precision, Brier и precision/recall при бюджете 10%. Модель обучена на будущих доставленных заказах, а не на полном потоке всех оплат; это ограничение для внедрения.'''),
        code("""from src.model import FEATURES, NUMERIC, CATEGORICAL, make_pipeline
print('Числовые признаки:', NUMERIC)
print('Категориальные признаки:', CATEGORICAL)
display(r['model_selection'])
display(r['model_metrics'])
display(pd.Series(r['model_card']['split_counts']).to_frame('orders'))
print('Выбранный C:', r['model_card']['chosen_C_by_july_AP'])
print('Пайплайн: median -> log1p -> scaling; imputation -> one-hot; LogisticRegression')
display(Image(filename=str(ROOT/'reports/figures/07_model_precision_recall.png')))"""),
        md('''На августе модель лучше ранжирует риск, но её Brier score хуже константной базы. Это ограничивает использование вероятностей без отдельной калибровки. Результат по одному месяцу не гарантирует стабильность на следующих месяцах.

Top-10% рассчитан ретроспективно внутри тестового месяца. Для онлайн-применения порог надо задать по прошлым данным или использовать отдельную политику очереди; текущий расчёт показывает только сценарий мощности команды.'''),
        md('''## 11. Дизайн будущего A/B-теста

Изменение: прозрачный статус/уведомление для заранее зафиксированного риск-сегмента. Рандомизация по покупателю, один первый пригодный заказ в основном анализе.

Основная метрика — обращение в поддержку по доставке в фиксированное окно до обещанной даты + 30 дней, знаменатель — все рандомизированные покупатели. Guardrails: отмены, доставка, отписки, стоимость поддержки. Оценки и доля ответов на опрос — дополнительные показатели.

**В Olist нет обращений в поддержку.** Ни baseline, ни эффект этой метрики не измерены. Ниже только сценарии планирования: baseline 5/10/20%, MDE 2 п.п., alpha 5%, power 80%, две равные группы. Перед запуском нужны реальные baseline и трафик риск-сегмента.'''),
        code("""display(r['ab_planning'])
print('Пример: при предположении 10% -> 8% нужно 3213 покупателей на группу.')
print('Это плановый расчёт, не результат A/B-теста и не число дней теста.')"""),
        md('''## 12. Вывод и ограничения

1. Опоздание связано с более высокой долей низких оценок; определение и момент отзыва существенно влияют на размер связи.
2. Маршруты и штаты различаются по срокам, но из сравнения не следует причинный эффект логистического изменения.
3. Риск-модель даёт основу для выделения сегмента; её калибровка и переносимость ещё не доказаны.
4. Приоритетная продуктовая идея требует отдельного эксперимента и нового логирования.

Нет данных для CR сайта, выручки от изменения, CAC, причинного retention uplift и фактической стоимости поддержки. Наблюдательные данные, пропуски, отбор ответивших и общие продавцы ограничивают обобщение.

Полный отчёт: `reports/summary.html` или `reports/summary.md`. Как защищать проект: `docs/interview_guide.md`. Краткое описание для резюме: `docs/resume_bullets.md`.'''),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
    })
    target = ROOT / 'notebooks/01_product_analysis.ipynb'
    nbf.write(notebook, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--in-process', action='store_true',
                        help='Execute via IPython without opening kernel sockets')
    args = parser.parse_args()
    path = build()
    if args.execute:
        notebook = nbf.read(path, as_version=4)
        if args.in_process:
            # Suitable for environments that forbid TCP kernel sockets. Code
            # actually runs; rich displays and streams become normal outputs.
            import os
            from IPython.core.interactiveshell import InteractiveShell
            from IPython.utils.capture import capture_output
            shell = InteractiveShell.instance()
            old_cwd = Path.cwd()
            try:
                os.chdir(ROOT)
                count = 0
                for cell in notebook.cells:
                    if cell.cell_type != 'code':
                        continue
                    count += 1
                    with capture_output(stdout=True, stderr=True, display=True) as captured:
                        result = shell.run_cell(cell.source, store_history=True)
                    error = result.error_before_exec or result.error_in_exec
                    if error is not None:
                        raise RuntimeError(f'Notebook code cell {count} failed') from error
                    outputs = []
                    if captured.stdout:
                        outputs.append(nbf.v4.new_output('stream', name='stdout', text=captured.stdout))
                    if captured.stderr:
                        outputs.append(nbf.v4.new_output('stream', name='stderr', text=captured.stderr))
                    for item in captured.outputs:
                        outputs.append(nbf.v4.new_output('display_data', data=item.data,
                                                         metadata=item.metadata))
                    cell.execution_count, cell.outputs = count, outputs
                notebook.metadata['execution_method'] = 'IPython in-process; all code cells executed'
            finally:
                os.chdir(old_cwd)
        else:
            from nbclient import NotebookClient
            client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                                    resources={'metadata': {'path': str(ROOT)}})
            client.execute()
        nbf.write(notebook, path)
        print('Executed all cells successfully:', path.name)
    else:
        print('Built notebook:', path.name)


if __name__ == '__main__':
    main()
