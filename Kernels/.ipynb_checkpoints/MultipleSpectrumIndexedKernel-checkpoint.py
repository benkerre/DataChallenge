import time
import numpy as np
from Kernels.SpectrumKernelIndexed import *



class MultipleSpectrumKernel:

    def __init__(self, KList):
        """
        KList is the list of k
        kernels is all considered SpectrumKernelIndexed instance associeted to Klist
        """  
        self.KList = KList
        self.kernels = [SpectrumKernelIndexed(k) for k in self.KList]
    
    def evaluate(self, x, y):
        """
        Returns the k(sequence1, sequence2).
        """ 
        evale = 0
        for kernel in self.kernels:
            evale += kernel.evaluate(x, y)   
        return evale
    
    def GramMatrix(self, Xtrain):

        print("Compute Multiple Spectrum Kernel Train")
        start = time.time()

        n_samples = Xtrain.shape[0]
        GramMatrix = np.zeros((n_samples, n_samples))
        for kernel in self.kernels:
            GramMatrix += kernel.GramMatrix(Xtrain)
        
        end = time.time()
        print("Calculation time:", "{0:.2f}".format(end - start))
        return GramMatrix

    def compute_KTest(self, Xtrain, Xtest):

        print("Called Multiple Spectrum Kernel Test")
        start = time.time()
        
        n_samples, n_features = Xtest.shape[0], Xtrain.shape[0]
        K = np.zeros((n_samples, n_features))
        for kernel in self.kernels:
            K += kernel.compute_KTest(Xtrain, Xtest)
        
        end = time.time()
        print("Calculation time:", "{0:.2f}".format(end - start))
        return K



