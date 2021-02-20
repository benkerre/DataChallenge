import numpy as np
from tqdm import tqdm as tqdm
from itertools import product

# K-M Mismatch Kernel

class KMMismatch:
    
    def _init__(self, k=3, m=1):
        self.k = k
        self.m = m
        self.kmersList = self.make_kmer()

    def make_kmer(self):
        """
        Return all possible k-mer of {'A','C','T','G'}
        """           
        return [''.join(kmer) for kmer in product('ACGT', repeat=self.k)]


    def features_map_sequence_mismatch_kmer(kmer):
        """
        Map the kmer to the feature space
        """
        feature_map_kmer = []
        for kmers in self.kmersList:
            mismatch = sum([(kmers[i]!=kmer[i]) for i in range(self.k)])
            feature_map_kmer.append(1*(mismatch <= self.m))
        return np.array(feature_map_kmer)
    
    def features_map_sequence_mismatch_sequence(sequence):
        """
        Map the sequence to the feature space
        """
        feature_map_sequence = np.repeat(0,4**self.k)
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            feature_map_sequence += self.features_map_sequence_mismatch_kmer(kmer)
        return feature_map_sequence


    def evaluate(sequence1, sequence2):
        """
        Compute K(x,y)
        """
        return self.features_map_sequence_mismatch_sequence(sequence1).dot(self.features_map_sequence_mismatch_sequence(sequence2))
    
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

