# Multimodal Learning With Categorical Embedding

## Todos
1. Get stats for each feature 
2. Extract selected / unselected features
3. Train full models with features, with selected features and without features
5. Visualize embeddings
6. Write report



## Segmentation models

### Pointrend classes
```
['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer', 'toothbrush'] 
```


## Experiments

General setting: consistent only, with transform.

Top 20 elems

|                                      | svm score            | average absolute correlation | max absolute correlation | uninformed guess   | performance increase task1 | performance increase task 2 |
| ------------------------------------ | -------------------- | ---------------------------- | ------------------------ | ------------------ | -------------------------- | --------------------------- |
| seg parse task1 (raw, not selection) | 0.75065097385689     | 0.039936896992649754         | 0.2671258520103224       | 0.6608686595146339 |                            |                             |
| seg parse task2 (raw, not selection) | 0.6856023506366308   | 0.03339309080332272          | 0.3145600649171777       | 0.5308521057786484 |                            |                             |
| seg parse task1 (selected)           | 0.7400270805124466   | 0.1631294098247213           | 0.2671258520103224       | 0.6608686595146339 |                            |                             |
| seg parse task2 (selected)           | 0.6622592229840026   | 0.18025121919956205          | 0.3145600649171777       | 0.5308521057786484 |                            |                             |
| is raw task1                         | (0.6629517758566816) | 0.17093582817624034          | 0.17093582817624034      | 0.6608686595146339 |                            |                             |
| is raw task2                         | (0.5308521057786484) | 0.046634066630442964         | 0.09947684241862445      | 0.5308521057786484 |                            |                             |
| is text task1                        | (0.6608686595146339) | 0.08242194982785796          | 0.08242194982785796      | 0.6608686595146339 |                            |                             |
| is text task2                        | (0.5308521057786484) | 0.06754924277721482          | 0.1054613934799101       | 0.5308521057786484 |                            |                             |

Used all training samples to evaluate extractors and select features



## Observations

Vehicle damage may be correctly predicted even if it is low-resource.