import numpy as np
from Kernels import *
from Algorithm import *



print("********* Training on the 3 learning sets *********\n")

################################################################################################################################################

print(">> Read Train set n°0")

Xtrain0 = np.loadtxt('./Data/Xtr0.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)
Ytrain0 = np.loadtxt('./Data/Ytr0.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(int)
Ytrain0 = 2*Ytrain0-1

#GramMatrix0 = np.loadtxt('./Xtrain0-Multiple-Spectrum-Kernel-[1-10]', skiprows=1, usecols=range(1,2001), dtype=str, delimiter=',').astype(float)
#svm0 = RegularizedSvm(regularization_rate=0.1) 
#svm0.fit(Xtrain0, Ytrain0, GramMatrix=GramMatrix0)
Kernel0 = MultipleSpectrumIndexedKernel(KList=range(1,11))
svm0 = RegularizedSvm(kernel=Kernel0, regularization_rate=0.1)
svm0.fit(Xtrain0, Ytrain0)
svm0.classification_report()

print("\n")

################################################################################################################################################

print(">> Read Train set n°1")

Xtrain1 = np.loadtxt('./Data/Xtr1.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)
Ytrain1 = np.loadtxt('./Data/Ytr1.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(int)
Ytrain1 = 2*Ytrain1-1

#GramMatrix1 = np.loadtxt('./Xtrain1-Multiple-Spectrum-Kernel-[1-10]', skiprows=1, usecols=range(1,2001), dtype=str, delimiter=',').astype(float)
#svm1 = RegularizedSvm(regularization_rate=0.1) 
#svm1.fit(Xtrain1, Ytrain1, GramMatrix=GramMatrix1)
Kernel2 = MultipleSpectrumIndexedKernel(KList=range(1,11))
svm1 = RegularizedSvm(kernel=Kernel2, regularization_rate=0.1)
svm1.fit(Xtrain1, Ytrain1)
svm1.classification_report()

print("\n")

################################################################################################################################################

print(">> Read Train set n°2")

Xtrain2 = np.loadtxt('./Data/Xtr2.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)
Ytrain2 = np.loadtxt('./Data/Ytr2.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(int)
Ytrain2 = 2*Ytrain2-1

#GramMatrix2 = np.loadtxt('./Xtrain2-Multiple-Spectrum-Kernel-[1-10]', skiprows=1, usecols=range(1,2001), dtype=str, delimiter=',').astype(float)
#svm2 = RegularizedSvm(regularization_rate=0.1) 
#svm2.fit(Xtrain2, Ytrain2, GramMatrix=GramMatrix2)
Kernel2 = MultipleSpectrumIndexedKernel(KList=range(1,11))
svm2 = RegularizedSvm(kernel=Kernel2, regularization_rate=0.1)
svm2.fit(Xtrain2, Ytrain2)
svm2.classification_report()

print("\n")

################################################################################################################################################

print("********* Load Test sets and predict *********\n")

Xtest0 = np.loadtxt('./Data/Xte0.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)
Xtest1 = np.loadtxt('./Data/Xte1.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)
Xtest2 = np.loadtxt('./Data/Xte2.csv', skiprows=1, usecols=(1,), dtype=str, delimiter=',').astype(str)

Ytest0 = svm0.predict(Xtest0)
Ytest1 = svm1.predict(Xtest1)
Ytest2 = svm2.predict(Xtest2)

# Map {-1, 1} back to {0, 1} (we can use (Ytest0+1)/2 to map in {0,1}, and cast the result in int representation)
Ytest0[Ytest0 == -1] = 0
Ytest1[Ytest1 == -1] = 0
Ytest2[Ytest2 == -1] = 0

# Concatenate Yests 0-1-2
AllYtest = np.concatenate((Ytest0,Ytest1,Ytest2))

print("********* Generating Yest.csv file *********\n")

Ytest = open('Ytest.csv', 'w')
Ytest.write("Id,Bound\n")
for idx, pred in enumerate(AllYtest):
        Ytest.write(str(idx)+","+str(pred)+"\n")
Ytest.close()