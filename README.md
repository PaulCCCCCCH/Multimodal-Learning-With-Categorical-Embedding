# Multimodal Learning With Categorical Embedding

90min 120min
## Ideas
1. Use a rule-based system to select from pre-defined category labels.
2. 


## Pipeline: Decision Tree
1. Read data for training set
2. Pre-process each sample
   1. Is this image a text?
   2. Is this image a 
3. Save as numpy



## Experiment1
- Use OCR to see if image has text, without appending to the text.
- Look at pixels to decide if it is a natural image or not.
- Use clustering/instance parsing to generate further labels

## Experiment2
- Apply word-frequency info
- Use semantic parsing to generate further labels.



## Segmentation models

### Pointrend classes
```
['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer', 'toothbrush'] 
```

