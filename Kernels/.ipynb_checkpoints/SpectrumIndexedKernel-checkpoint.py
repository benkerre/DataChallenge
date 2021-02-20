import time
import numpy as np
import scipy.sparse as sparse


class SpectrumIndexedKernel:
 
    def __init__(self, k):
        self.k = k
        self.nonzeroscolumn = None
        self.alphabet = {"A":0, "C":1, "G":2, "T":3}

    def valueOfKmer(self, kmer):
        """
        Returns the value of kmer according to the alphabet dictionary.
        Example: If k=3, valueOfKmer('ATC') = 0*(4^0) + 3*(4^1) + 1*(4^2) = 21
        This function is an injection, in fact, if valueOfKmer(kmer1=abc) = valueOfKmer(kmer2=efg) --> 1*a+4*b+c*16 = 1*e+4*f+16*g --->
        ---> a/16 + b/4 + c = a/16 + b/4 + c ---> c=g because a/16 + b/4 < 3/16 + 3/4 = 15/16 < 1, an same, b=f and a=e.
        This function will be very useful to calculate the frequency of kmers in a sequence and in a very fast way compared to the naive method
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
        # Browse all kmers of sequence
        for idx in range(len(sequence) - self.k + 1):
            kmer = sequence[idx:(idx+self.k)]
            # phisequence[val] is the number of times kmer appears in sequence, because valueOfKmer is injective
            phiofsequence[self.valueOfKmer(kmer)] += 1
        return phiofsequence
    
    def evaluate(self, sequence1, sequence2):
        """
        Returns the k(sequence1, sequence2).
        """ 
        return self.phi(sequence1).dot(self.phi(sequence2))

    def SparseGramMatrix(self, Xtrain):
        
        """
        Compute the Gram Matrix intelligently, without using evaluate with time complexity: O(2k(L-k+1)+4^k)
        """ 
        
        # Convert vector of all sequences to a matrix of number according to alphabet dictionary, following rule: {A:0,C:1,G:2,T:3}
        SeqSize = len(Xtrain[0]); n_samples = Xtrain.shape[0]
        X_encoded_to_num_vector = np.zeros((n_samples, SeqSize), dtype=np.uint8)
        for i in range(n_samples):
            for j in range(SeqSize):
                X_encoded_to_num_vector[i][j] = self.alphabet[Xtrain[i][j]]

        # Define the matrix which allows after multiplication with X_encoded_to_num_vector, to compute the value of the kmers of each sequence in Xtrain      
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
 
        print("Compute Spectrum Indexed Kernel for XTrain with k=",self.k)
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
            
        print("Compute Spectrum Indexed Kernel K(Xtest,Xtrain)")
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

