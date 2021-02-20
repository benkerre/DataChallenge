import numpy as np
from Algorithms import *
from Kernels import *
from sklearn.model_selection import train_test_split


#########################
# Tuning alpha parameter
#########################


print("Load Dataset n°0 ")
Xtrain2 = np.loadtxt('./Data/Xtr2.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)
Ytrain2 = np.loadtxt('./Data/Ytr2.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(int)
Ytrain2 = 2*Ytrain2-1

# Split Data to train and test sets
Xtr, Xte, Ytr, Yte = train_test_split(Xtrain2, Ytrain2, test_size=0.2)

# Tuning
klist = [1,2,3,4,5,6,7,8,9,10]; alpha = 0.1; train_scores = [], test_scores = []
for i in range(1,11):
    kernel = MultipleSpectrumIndexedKernel(KList=klist[:i])
    svm = RegularizedSvm(kernel=kernel,alpha=alpha)
    svm.fit(Xtr, Ytr)
    print("klist:", klist[:i])
    pred = svm.predict(Xte)
    tmp = Yte == pred
    accuracy_test = np.sum(tmp) / np.size(tmp)
    test_scores.append(accuracy_test)
    pred = np.sign(svm.GramMatrix.dot(svm.solution.reshape((svm.solution.size, 1)))).reshape(-1)
    tmp = Ytr == pred
    accuracy_train = np.sum(tmp) / np.size(tmp)
    
# PLot 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle('Train and Test accuracy = f(List of k)')
ax1.plot(alphas,train_scores)
ax2.plot(alphas,test_scores)
ax1.set_ylabel('Train accuracy')
ax1.set_xlabel('List of k')
ax1.set_title('Train accuracy = f(List of k)')
ax2.set_ylabel('Test accuracy')
ax2.set_xlabel('List of k')
ax2.set_title('Test accuracy = f(List of k)')
