# Fake News Detection

## Dataset

### KDD2020

[Fake News Detection Challenge KDD 2020](https://www.kaggle.com/competitions/fakenewskdd2020/overview)

- train: 3491
- validation: 997
- test: 499

### GonzaloA/fake_news

[fake_news](https://huggingface.co/datasets/GonzaloA/fake_news)

- train: 24353
- validation: 8117
- test: 8117

## Results

### Fine-tuning on KDD2020

| Model                   | Accuracy           | F1                 | Loss                |
| ----------------------- | ------------------ | ------------------ | ------------------- |
| distilbert-base-uncased | 0.7655310621242485 | 0.7673580407434882 | 0.46802181005477905 |

### Fine-tuning on GonzaloA/fake_news

| Model                   | Accuracy           | F1                 | Loss                 |
| ----------------------- | ------------------ | ------------------ | -------------------- |
| bert-base-uncase        | 0.9882961685351731 | 0.9882994966274701 | 0.026664618402719498 |
| distilbert-base-uncased | 0.986694591597881  | 0.9866977733534859 | 0.029809903353452682 |
| roberta-base            | 0.986694591597881  | 0.9866872535155707 | 0.024871505796909332 |

### Few Shot Learning on KDD2020

### Few Shot Learning on GonzaloA/fake_news
