# CaseLab ML — ДЗ №3

Готовое решение для запуска в Google Colab.

## Структура

- `06_cnn_classification/assignment.ipynb` — исходный шаблон задания.
- `06_cnn_classification/solution.ipynb` — решение классификации CNN.
- `07_cnn_transfer_segmentation/assignment.ipynb` — исходный шаблон задания.
- `07_cnn_transfer_segmentation/solution.ipynb` — решение transfer learning и сегментации.

## Запуск

1. Открыть `solution.ipynb` в Google Colab.
2. Убедиться, что при необходимости выбран GPU.
3. Для быстрой проверки использовать `FAST_MODE = True`.
4. Для финального запуска использовать `FAST_MODE = False`.
5. Выполнить `Runtime -> Run all`.

Результаты экспериментов, графики и выводы формируются по фактически полученным данным; заранее заданных метрик в ноутбуках нет.

## Основные технические решения

- фиксированный seed и воспроизводимый split;
- отдельные transforms для train/validation/test;
- единый pipeline обучения и оценки;
- автоматический выбор CPU/GPU;
- mixed precision на CUDA;
- сохранение лучших checkpoint'ов;
- очистка памяти между тяжёлыми экспериментами;
- обработка нестабильных случаев метрик и размерностей;
- синхронные геометрические transforms для изображения и segmentation mask;
- реальный пересчёт accuracy и IoU после обучения.
