import time
import numpy as np
from itertools import product
import scipy.sparse as sparse



########################################################################
### SpectrumKernelPreindexed                                                         
########################################################################

class SpectrumKernelPreindexed:
 
    def __init__(self, k):
        self.k = k
        self.nonzeroscolumn = None
        self.alphabet = {"A":0, "C":1, "G":2, "T":3}

    def valueOfKmer(self, kmer):
        """
        Returns the value of kmer according to the alphabet dictionary.
        Example: If k=3, valueOfKmer('ATC') = 0*(4^0) + 3*(4^1) + 1*(4^2) = 21
        """   
        valueOfKmer = 0
        for i in range(self.k):
            valueOfKmer += self.alphabet[kmer[i]] * (4**i)
        return valueOfKmer

    def phi(self, sequence):  
        """
        Returns the map vector of sequence.
        """ 
        dimFeaturesSpace = 4**self.k
        phiofsequence = np.zeros(dimFeaturesSpace, dtype=np.uint16)
        # Extract all kmers of sequence
        for idx in range(len(sequence) - self.k + 1):
            kmer = sequence[idx:(idx+self.k)]
            # phisequence[val] is the number of kmer of valueOfKmer equal to val
            phiofsequence[self.valueOfKmer(kmer)] += 1
        return phiofsequence
    
    def evaluate(self, sequence1, sequence2):
        """
        Returns the k(sequence1, sequence2).
        """ 
        return self.phi(sequence1).dot(self.phi(sequence2))

    def SparseGramMatrix(self, Xtrain):
        
        # Convert vector of all sequences to a matrix of number according to alphabet dictionary
        SeqSize = len(Xtrain[0]); n_samples = Xtrain.shape[0]
        X_encoded_to_num_vector = np.zeros((n_samples, SeqSize), dtype=np.uint8)
        for i in range(n_samples):
            for j in range(SeqSize):
                X_encoded_to_num_vector[i][j] = self.alphabet[Xtrain[i][j]]

        MatrixKmersIdentifier = np.zeros((SeqSize - self.k + 1, SeqSize), dtype=np.uint32)
        for i in range(self.k):
            Iemediag = np.diagonal(MatrixKmersIdentifier, i)
            Iemediag.setflags(write=True)
            Iemediag.fill(4**i)
            
        MatrixKmersIdentifierInX = X_encoded_to_num_vector.dot(MatrixKmersIdentifier.T)
        SparseGramMatrix = sparse.lil_matrix((n_samples, 4 ** self.k), dtype=np.float64)
        for i in range(MatrixKmersIdentifierInX.shape[0]):
            res = np.bincount(MatrixKmersIdentifierInX[i, :])
            SparseGramMatrix[i, :res.size] += res
            
        return SparseGramMatrix

    def GramMatrix(self, Xtrain):
 
        print("Compute Spectrum Kernel Train Preindexed with k=",self.k)
        start = time.time()

        self.SGramMatrix = self.SparseGramMatrix(Xtrain)
        # Remove zero columns
        self.nonzeroscolumn = self.SGramMatrix.getnnz(0)>0
        self.SGramMatrix = self.SGramMatrix[:,self.nonzeroscolumn]
        if sparse.isspmatrix(self.SGramMatrix):
            self.SGramMatrix = self.SGramMatrix.todense()
        self.GramMatrix = np.array(self.SGramMatrix.dot(self.SGramMatrix.T), dtype=np.float64)
        
        end = time.time()
        print("Calculation time:", "{0:.2f}".format(end - start))  
        return self.GramMatrix
        

    def compute_KTest(self, Xtrain, Xtest):
            
        print("Called SpectrumKernelPreindexed.compute_K_test")
        start = time.time()

        n_samples, n_features = Xtest.shape[0], Xtrain.shape[0]
        SparseGramMatrix = self.SparseGramMatrix(Xtest) 
        SparseGramMatrix = SparseGramMatrix[:,self.nonzeroscolumn]
        if sparse.isspmatrix(SparseGramMatrix):
            SparseGramMatrix = SparseGramMatrix.todense()
        K = np.array(SparseGramMatrix.dot(self.SGramMatrix.T), dtype=np.float64)

        end = time.time()
        print("Calculation time:", "{0:.2f}".format(end - start))
        return K


########################################################################
### MultipleSpectrumKernel                                                         
########################################################################


class MultipleSpectrumKernel:

    def __init__(self, KList):
  
        self.KList = KList
        self.kernels = [SpectrumKernelPreindexed(k) for k in self.KList]
    
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



