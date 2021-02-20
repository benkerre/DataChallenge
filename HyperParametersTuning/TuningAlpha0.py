import numpy as np
from Algorithms import *
from Kernels import *
from sklearn.model_selection import train_test_split


#########################
# Tuning alpha parameter
#########################


print("Load Dataset n°1 ")
Xtrain0 = np.loadtxt('./Data/Xtr0.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)
Ytrain0 = np.loadtxt('./Data/Ytr0.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(int)
Ytrain0 = 2*Ytrain0-1

# Split Data to train and test sets
Xtr, Xte, Ytr, Yte = train_test_split(Xtrain0, Ytrain0, test_size=0.2)

# Tuning Alpha for Xtrain0
alphas = np.linspace(0.1,1,10); test_scores = []; train_scores = []
for alpha in alphas:
    svm = RegularizedSvm(kernel=MultipleSpectrumIndexedKernel(KList=range(1,11)),alpha=alpha) 
    svm.fit(Xtr, Ytr)
    print("Alpha:", alpha)
    # For test
    pred = svm.predict(Xte)
    tmp = Yte == pred
    accuracy = np.sum(tmp) / np.size(tmp)
    test_scores.append(accuracy)
    # For train
    pred = np.sign(svm.GramMatrix.dot(svm.solution.reshape((svm.solution.shape[0], 1)))).reshape(-1)
    tmp = Ytr == pred
    accuracy_train = np.sum(tmp) / np.size(tmp)
    train_scores.append(accuracy)
    
# PLot 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle('Train and Test accuracy = f(Alpha)')
ax1.plot(alphas,train_scores)
ax2.plot(alphas,test_scores)
ax1.set_xlabel('Aplhas')
ax1.set_ylabel('Test score')
ax1.set_title('Train accuracy = f(Alpha)')
ax1.legend(["Best Alpha is:"+str(alphas[np.argmax(test_scores)])])
ax2.set_xlabel('Aplhas')
ax2.set_ylabel('Test score')
ax2.set_title('Test accuracy = f(Alpha)')
ax2.legend(["Best Alpha is:"+str(alphas[np.argmax(train_score)])])
