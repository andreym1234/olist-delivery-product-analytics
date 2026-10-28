# Публикация отдельного проекта

Репозитории, соответствующие скриншоту резюме и проверенные по README:

- https://github.com/andreym1234/ecommerce-conversion-analysis
- https://github.com/andreym1234/online-shoppers-purchase-classification

Этот проект подготовлен как отдельная папка `olist-delivery-product-analytics`; он не изменяет существующие репозитории. Новый репозиторий в аккаунте ещё не создан.

1. Распакуй архив и прочитай README. Выполненный ноутбук, графики и отчёты уже внутри.
2. Разберись с расчётами и внеси свои дополнения/заметки. Пример полезного дополнения — заранее сформулированный вопрос к одному сегменту discovery и его аккуратный разбор.
3. Создай на GitHub пустой репозиторий `olist-delivery-product-analytics` без автоматического README.
4. Из корня распакованной папки выполни:

```bash
git init -b main
git add .
git commit -m "Add Olist delivery experience research"
git remote add origin https://github.com/andreym1234/olist-delivery-product-analytics.git
git push -u origin main
```

Перед коммитом `git status --short` должен показывать код, документацию, готовые графики, таблицы и ноутбук. Исходные CSV, SQLite, окружение и служебные файлы исключены через .gitignore.

Краткое описание репозитория:

> Product analysis of Olist delivery experience: SQL mart, EDA, hypothesis testing, review-selection sensitivity, late-delivery risk model and experiment design.

Подходящие темы GitHub: `product-analytics`, `eda`, `sql`, `python`, `statistics`, `hypothesis-testing`, `ecommerce`, `olist`.

После публикации добавь настоящую ссылку в резюме. Верифицированные формулировки — в docs/resume_bullets.md. Данные и производные результаты сохраняют атрибуцию Olist; см. DATA_LICENSE.md.
