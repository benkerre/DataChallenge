import numpy as np
from tqdm import tqdm as tqdm
from itertools import product

# spectrum kernel 

class SpectrumKernel:
    
    def _init__(self, k=3):
        self.k = k

    def make_kmer(self):
        """
        Return all possible k-mer of {'A','C','T','G'}
        """           
        return [''.join(kmer) for kmer in product('ACGT', repeat=self.k)]


    def features_map_sequence_spectrum(sequence, k, kmers_list):
        """
        Map the sequence to the feature space
        """
        feature_map_sequence = np.repeat(0,4**k)
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            for idx, e in enumerate(kmers_list):
                feature_map_sequence[idx] += (e == kmer)         
        return feature_map_sequence
    
    def evaluate(sequence1, sequence2):
        """
        Compute K(x,y)
        """
        return self.feature_map_sequence(sequence1).dot(self.feature_map_sequence(sequence2))
    
    def GramMatrix(self, Xtrain):
        """
        Compute Gram Matrix
        """
        self.Xtrain = Xtrain
        n_samples = Xtrain.shape[0]
        self.GramMatrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                self.GramMatrix[i, j] = self.GramMatrix[j, i]  = evaluate(Xtrain[i], Xtrain[j])              
        return GramMatrix
    
    def compute_KTest(self, Xtest):
        """
        Compute Ktest
        """
        n_samples, n_features = Xtest.shape[0], Xtrain.shape[0]
        Ktest = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_features):
                Ktest[i, j] = Ktest[j, i]  = evaluate(Xtest[i], self.Xtrain[j])              
        return Ktest