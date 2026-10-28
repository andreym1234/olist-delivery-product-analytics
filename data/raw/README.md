# Исходные данные

Запусти из корня проекта:

```bash
python scripts/download_data.py
```

Или скачай архив с https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce и передай его через `--archive`.

CSV не включены в git. В data/source_manifest.json записаны контрольные суммы восьми использованных исходных файлов. Географическая таблица загружается вместе с архивом, но в исследовании не используется.
