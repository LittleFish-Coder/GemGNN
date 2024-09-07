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

### Zero Shot Classification
|Model| Dataset | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
|bart-large-mnli|KDD2020|0.6232|0.6699|0.6232|0.5285|
|bart-large-mnli|GonzaloA/fake_news|0.5461|0.5899|0.5461|0.4210|

### Fine-tuning on KDD2020

| Model                   | Accuracy           | F1                 | Loss                |
| ----------------------- | ------------------ | ------------------ | ------------------- |
| bert-base-uncase        | 0.7835671342685371 | 0.7807691946073421 | 0.46070271730422974 |
| distilbert-base-uncased | 0.7655310621242485 | 0.7673580407434882 | 0.46802181005477905 |
| roberta-base            | 0.8156312625250501 | 0.8125283885140466 | 0.4082072079181671  |

### Fine-tuning on GonzaloA/fake_news

| Model                   | Accuracy           | F1                 | Loss                 |
| ----------------------- | ------------------ | ------------------ | -------------------- |
| bert-base-uncase        | 0.9882961685351731 | 0.9882994966274701 | 0.026664618402719498 |
| distilbert-base-uncased | 0.986694591597881  | 0.9866977733534859 | 0.029809903353452682 |
| roberta-base            | 0.986694591597881  | 0.9866872535155707 | 0.024871505796909332 |

### Few Shot Learning on KDD2020

| Samples | Model                   | Accuracy           | F1                 | Loss               |
| ------- | ----------------------- | ------------------ | ------------------ | ------------------ |
| 10      | distilbert-base-uncased | 0.5591182364729459 | 0.5447740312435575 | 0.6902174949645996 |
| 100     | distilbert-base-uncased | 0.591182364729459  | 0.4392916815999757 | 0.677597165107727  |

### Few Shot Learning on GonzaloA/fake_news

| Samples | Model                   | Accuracy               | F1                 | Loss               |
| ------- | ----------------------- | ---------------------- | ------------------ | ------------------ |
| 10      | distilbert-base-uncased | 0.48367623506221513    | 0.3341564119711251 | 0.6532924771308899 |
| 100     | distilbert-base-uncased | **0.9275594431440187** | 0.9273481720070647 | 0.5080302953720093 |

### Model Generalization (Cross-dataset)

We inspect the generalization of the models trained on one dataset and tested on another datase, the accuracy is used as the evaluation metric.

- Train on GonzaloA/fake_news and test on KDD2020
  | Model | Train Sample|In-dataset (GonzaloA/fake_news) | cross-dataset KDD2020|
  | --- | --- | --- | --- |
  | bert-base-uncase | full |0.9882961685351731 | 0.42685370741482964 |
  | distilbert-base-uncased |full | 0.986694591597881 | 0.5150300601202404 |
  | roberta-base | full |0.986694591597881 | 0.503006012024048 |
  | distilbert-base-uncased | 10 | 0.48367623506221513 | 0.4088176352705411 |
  | distilbert-base-uncased | 100 | 0.9275594431440187 | 0.46292585170340683 |

- Train on KDD2020 and test on GonzaloA/fake_news

  | Model                   | Train Sample | In-dataset (KDD2020) | cross-dataset GonzaloA/fake_news |
  | ----------------------- | ------------ | -------------------- | -------------------------------- |
  | bert-base-uncase        | full         | 0.7835671342685371   | 0.37058026364420354              |
  | distilbert-base-uncased | full         | 0.7655310621242485   | 0.42540347418997165              |
  | roberta-base            | full         | 0.8156312625250501   | 0.7220648022668473               |
  | distilbert-base-uncased | 10           | 0.5591182364729459   | 0.3767401749414808               |
  | distilbert-base-uncased | 100          | 0.591182364729459    | 0.5340643094739436               |
