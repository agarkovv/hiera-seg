# Project: Hierarchical semantic segmentation

## Literature

**SOTA:** https://arxiv.org/abs/2203.14335

Pointcare ball space, та же задача: https://arxiv.org/abs/2404.03778

**Human part segmentation:**

- https://arxiv.org/abs/1912.09622
- https://arxiv.org/abs/1803.05675
- https://arxiv.org/abs/2001.06804
- https://arxiv.org/abs/2003.04845
- https://arxiv.org/abs/2103.04570
- https://aaai.org/papers/12336-progressive-cognitive-human-parsing/

## Backbones for segmentation

- [DeepLabV3+](https://arxiv.org/abs/1802.02611v3)
- [MaskFormer](https://proceedings.neurips.cc/paper_files/paper/2021/file/950a4152c2b4aa3ad78bdd6b366cc179-Paper.pdf)

![Screenshot 2024-07-16 at 12.06.52.png](Project%20Hierarchical%20semantic%20segmentation%2042633526ee6345e5b1f268304e1bc03c/Screenshot_2024-07-16_at_12.06.52.png)

## Approaches:

Я решил задачу, используя модель HSSN (SOTA для иерархической семантической сегментации), но с более легкими архитектурными решениями и более простой функцией потерь. Вот краткий обзор архитектуры:

### Model architecture:

1. Энкодер сегментации $f_{ENC}$ преобразует каждую картинку $I$ в признак $I \in \mathbb{R}^{H \times W \times C}$ . В качестве $f_{ENC}$ берется backbone модель.

_В своей реализации я использовал ResNet-50, но также можно рассмотреть следующие модели: HRNetV2-W48 или Transformer-based (i.e., Swin-Transformer)._

1. Голова сегментации $f_{SEG}$ преобразует $I$ в score-map $S \in \mathbb{R}^{H \times W \times |V|}$ для всех листовых классов в $V$. $V$ это все вершины в графе иерархии, e.g., “torso”, “upper leg”

_В своей реализации я использовал DeepLabV3+ модель. На скрине выше из статьи про MaskFormer видно, что она до 5 раз быстрее MaskFormer с небольшой просадкой качества в 0.5 mIoU. Также в качестве головы сегментора можно рассмотреть OCRNet._

### Loss:

Авторы DHSS используют модифицированный Focal-Loss для классификации. Он состоит из двух слагаемых:

1. Focal Tree-Min Loss:

$$
L^{FTM}(s) = \sum_{v \in V} - \hat{l_v}(1 - \min_{u \in A_v}(s_u))^{\gamma}\log(\min_{u \in A_v}(s_u)) - (1 - \hat{l_v})( \max_{u \in C_v}(s_u))^{\gamma}\log(1 - \max_{u \in C_v}(s_u))
$$

где

- $\gamma \geq 0$ это параметр для down-weighting (как в Focal Loss)
- $\hat{l_v}$ бинарный таргет принадлежности пикселя правильному пути от корня до листа (e.g. body - upper body - torso)
- $A_v, C_v$ это все предки и все дети вершины $v$
- $s_u$ - логиты принадлежности пикселя правильному пути от корня до листа

1. Tree-Triplet Loss

$$
L^{TT}(i, i^+, i^-) = \max(<i, i^+> - <i, i^-> + m, 0)
$$

Обычный трплет - лосс с якорем $i$ и двумя примерами - положительным и отрицательным. $m$ - margin.

_В своей реализации я использовал обычный Focal Loss. В следующей работе (про иерархическую сегментацию на шаре Пуанкаре) авторы в коде не используют FTT или TT Loss._

## Implementation details

![Screenshot 2024-07-17 at 21.41.26.png](Project%20Hierarchical%20semantic%20segmentation%2042633526ee6345e5b1f268304e1bc03c/Screenshot_2024-07-17_at_21.41.26.png)

![Screenshot 2024-07-17 at 21.41.49.png](Project%20Hierarchical%20semantic%20segmentation%2042633526ee6345e5b1f268304e1bc03c/Screenshot_2024-07-17_at_21.41.49.png)

![Screenshot 2024-07-17 at 22.36.18.png](Project%20Hierarchical%20semantic%20segmentation%2042633526ee6345e5b1f268304e1bc03c/Screenshot_2024-07-17_at_22.36.18.png)

- Mean root IoU (0) - метрика классификации body
- Mean middle IoU (1) - метрика классификации upper body и lower body
- Mean IoU - метрика классификации остальных частей тела

\*_При подсчете не учитываем background_

- Class IoU - IoU по классам (метки совпадают с метками из md файла с заданием)

## Как довести модель до идеала:

- Доучить побольше. Пока что я тренировал 1к из 30к шагов обучения (это ~8 часов обучения на M1 GPU)
- Добавить Tree Triplet Loss и Focal Tree Min Loss. Это немного замедлит обучение, т.к. каждый раз для логитов и таргетов придется обновлять граф (считать минимум или максимум) для каждого пикселя (т.е. $O(N \cdot H \cdot W)$, где $N$ - кол-во классов (у нас их 10), следующие множители это размер картинки)
- Попробовать перейти на шар Пуанкаре, чтобы улучшить топологию расположения кластеров:
  ![Screenshot 2024-07-17 at 22.13.59.png](Project%20Hierarchical%20semantic%20segmentation%2042633526ee6345e5b1f268304e1bc03c/Screenshot_2024-07-17_at_22.13.59.png)
- Использовать более тяжелые backbones (например, заменить ResNet50 на ResNet101)
- Использовать более тяжелые головы сегментора (например, MaskFormer, который я упоминал выше)

````

### 2 Запуск

```bash
python3 main.py --data_root="path/to/data" --batch_size=16 --enable_vis --loss_type=focal_loss --test_only
````
