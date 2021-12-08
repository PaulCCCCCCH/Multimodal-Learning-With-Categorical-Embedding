# Multimodal Learning With Categorical Embedding

Part of the idea for 11777 Group 3: Crisis Event Detection and Multimodal Image Classification


## To run
First setup the environment with 
```
pip install requirements.txt
```

Then, you can run training with commands in `train.sh` to train the model. Make sure to adjust the paths to fit your own setup.

To generate categorical embeddings, run `categorize.sh`.



## Feature correlations

|                                      | svm score            | average absolute correlation | max absolute correlation | uninformed guess   |
| ------------------------------------ | -------------------- | ---------------------------- | ------------------------ | ------------------ |
| seg parse task1 (raw, not selection) | 0.75065097385689     | 0.039936896992649754         | 0.2671258520103224       | 0.6608686595146339 |
| seg parse task2 (raw, not selection) | 0.6856023506366308   | 0.03339309080332272          | 0.3145600649171777       | 0.5308521057786484 |
| seg parse task1 (selected)           | 0.7400270805124466   | 0.1631294098247213           | 0.2671258520103224       | 0.6608686595146339 |
| seg parse task2 (selected)           | 0.6622592229840026   | 0.18025121919956205          | 0.3145600649171777       | 0.5308521057786484 |
| is raw task1                         | (0.6629517758566816) | 0.17093582817624034          | 0.17093582817624034      | 0.6608686595146339 |
| is raw task2                         | (0.5308521057786484) | 0.046634066630442964         | 0.09947684241862445      | 0.5308521057786484 |
| is text task1                        | (0.6608686595146339) | 0.08242194982785796          | 0.08242194982785796      | 0.6608686595146339 |
| is text task2                        | (0.5308521057786484) | 0.06754924277721482          | 0.1054613934799101       | 0.5308521057786484 |

Used all training samples to evaluate extractors and select features
