import time
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


class RegularizedSvm:

    def __init__(self, kernel=None,  regularization_rate=1):
        """
        Kernelized Support vector machines algorithm.
        Regularization_rate is regularization parameter
        """
        self.kernel = kernel
        self.regularization_rate = regularization_rate

    def fit(self, Xtrain, Ytrain):
        """
            Train the instance of algorithm on (Xtrain,Ytrain)
        """
        
        self.Xtrain, self.Ytrain = Xtrain, Ytrain    
        print("\nBuilding Gram Matrix")
        self.GramMatrix = self.kernel.GramMatrix(self.Xtrain)
        print("\nEnd Building")
        
        print("\nTraining SVM algorithm")
        start = time.time()
        n_samples = Xtrain.shape[0]
        P = matrix(self.GramMatrix, tc='d')
        q = matrix(-Ytrain, tc='d')
        G = matrix(np.append(np.diag(-Ytrain.astype(float)), np.diag(Ytrain.astype(float)), axis=0), tc='d')
        h = matrix(np.append(np.zeros(n_samples), np.ones(n_samples, dtype=float) / (2 * self.regularization_rate * n_samples), axis=0), tc='d')
        solution = solvers.qp(P, q, G, h)
        self.solution = np.array(solution['x'])
        end = time.time()
        print("End Solving. Calculation time:", "{0:.2f}".format(end - start))  
        
    
    def predict(self, Xtest):
        """
            Predict Yesy corresponding to Xtest
        """
        print("\nPredicting")
        KTest = self.kernel.compute_K_test(self.Xtrain, Xtest)
        Ytest = np.sign(KTest.dot(self.solution.reshape((self.solution.shape[0], 1))).reshape(-1)).astype(int)
        print("\nEnd Predicting")
        return Ytest  

    def classification_report(self):
        """
            Show the classification report of the instance on Training data and ROC/Recall-Precicon Curves
        """
        
        pred = np.sign(self.GramMatrix.dot(self.solution.reshape((self.solution.shape[0], 1)))).reshape(-1)
        print("\nClassification report on the Train set")
        print(classification_report(self.Ytrain, pred))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
        fig.suptitle('ROC and Recall-Precision Curves. \n(please close this window to continue running the program)\n')
        
        # ROC Curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(self.Ytrain, pred)
            roc_auc[i] = auc(fpr[i], tpr[i])
        auc_score = roc_auc_score(self.Ytrain, pred)
        ax1.plot(fpr[1], tpr[1])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver operating characteristic')
        ax1.legend(["AUC:{0:.2f}".format(auc_score)])
        
        # Recall Precision Curve
        precision, recall, thresholds = precision_recall_curve(self.Ytrain, pred)
        auc_precision_recall = auc(recall, precision)
        ax2.plot(recall, precision)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Recall precision curve')
        ax2.legend(["AUC:{0:.2f}".format(auc_precision_recall)])
        
        plt.show()
        