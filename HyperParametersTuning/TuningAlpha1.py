import numpy as np
from Algorithms import *
from Kernels import *
from sklearn.model_selection import train_test_split


#########################
# Tuning alpha parameter
#########################


print("Load Dataset n°1 ")
Xtrain1 = np.loadtxt('./Data/Xtr1.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)
Ytrain1 = np.loadtxt('./Data/Ytr1.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(int)
Ytrain1 = 2*Ytrain1-1

# Split Data to train and test sets
Xtr, Xte, Ytr, Yte = train_test_split(Xtrain1, Ytrain1, test_size=0.2)

# Tuning Alpha for Xtrain0
alphas = np.linspace(0.1,1,10); test_scores = []
for alpha in alphas:
    svm = RegularizedSvm(kernel=MultipleSpectrumKernel(KList=range(1,11)),alpha=alpha) 
    svm.fit(Xtr, Ytr)
    print("Alpha:", alpha)
    pred = svm.predict(Xte)
    tmp = Yte == pred
    accuracy = np.sum(tmp) / np.size(tmp)
    test_scores.append(accuracy)
    print("Test accuracy with alpha:", alpha, "is:", accuracy)
    
# PLot 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.plot(alphas,test_scores)
ax2.plot(alphas,train_score)
ax1.set_xlabel('Aplhas')
ax1.set_ylabel('Test score')
ax1.set_title('TestScore = f(Alpha)')
ax1.legend(["Best Alpha is:"+str(alphas[np.argmax(test_scores)])])
ax2.set_xlabel('Aplhas')
ax2.set_ylabel('Train score')
ax2.set_title('TrainScore = f(Alpha)')
ax2.legend(["Best Alpha is:"+str(alphas[np.argmax(train_score)])])