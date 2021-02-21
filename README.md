# DataChallenge

The purpose of this Data challenge is to predict whether a DNA sequence of the vocabulary {A,C,T,G} is binding site to a specific transcription factor or not. It's requested to implement machine learning algorithms based on kernel methods. For this purpose, a set of kernels for sequence classification were implemented and tested, using kernelized learning algorithms such as support vector machine. The results on the training and test sets were encouraging, indeed, we had an accuracy and recall of 99% on training set and  71.066% (9th rank) on the public leaderboard and 69.866% (9th rank) on the private leaderboard. (please click on this link [Data Challenge](https://www.kaggle.com/c/advanced-learning-models-2020/leaderboard))

## Execution

```bash
git clone git@github.com:benkerre/DataChallenge.git
```

```bash
cd DataChallenge
```

```bash
python3 start.py
```

## Results

Thanks to our implementation of the svm algorithm as well as to the different kernels to preprocess our data, we were able to answer the requested classification problem with a relatively high score of 71.066\% on the public Leader board. It is clear that this result can be improved by implementing other types of kernels, however, it requires computer resources large enough to be able to train the models for a long time without worrying about memory or blocking.

![Screenshot](Images/LEADERBOARD.png)
