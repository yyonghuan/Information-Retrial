# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:42:53 2017

@author: yangyonghuan
"""
import numpy as np
import scipy.sparse as sp
import numbers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.fixes import bincount
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity


class BM25Vectorizer(BaseException, TransformerMixin):
    
    def __init__(self, k1=1.5, b = 0.75, norm =None, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64):
        
        self.k1 = k1
        self.b = b
        self.norm = norm
        
        self.tf_vectorizer = CountVectorizer(input=input, 
                                             encoding=encoding,
                                             decode_error=decode_error, 
                                             strip_accents=strip_accents,
                                             lowercase=lowercase, 
                                             preprocessor=preprocessor, 
                                             tokenizer=tokenizer,
                                             stop_words=stop_words, 
                                             token_pattern=token_pattern,
                                             ngram_range=ngram_range, 
                                             analyzer=analyzer,
                                             max_df=max_df, 
                                             min_df=min_df, 
                                             max_features=max_features,
                                             vocabulary=vocabulary, 
                                             binary=binary, 
                                             dtype=dtype)
        

        
    def fit(self, X):
        """
        -	It should train the BM25 model on the given corpus docs
        -	Return nothing
        """
        X = self.tf_vectorizer.fit_transform(X).toarray()
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        n_samples, n_features = X.shape
        
        if sp.isspmatrix_csr(X):
            df = bincount(X.indices, minlength=X.shape[1])
        else:
            df = np.diff(sp.csc_matrix(X, copy=False).indptr)
        
        #compute idf weight
        #idf = np.log((float(n_samples)-df+0.5)/(df+0.5))
        idf = np.log(float(n_samples) / df) + 1.0
        self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, 
                                    n=n_features, format='csr')
        #compute the length for each document and average length of the corpus
        doc_len = np.sum(X,axis=1)
        self._doc_len = np.reshape(doc_len, (n_samples,1))
        self._avgdl = np.sum(X)/n_samples
        
    
    def transform(self, X):
        """
        -	It should convert the given corpus into the vector-space 
             representation based on the model we learn from ‘fit’ method
        -	Return a matrix where each row is one document, each column 
             is feature
        """
        X = self.tf_vectorizer.fit_transform(X).toarray()
        n_samples, n_features = X.shape
        
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64)
        
        expected_n_features = self._idf_diag.shape[0]
        if n_features != expected_n_features:
            raise ValueError("Input has n_features=%d while the model"" has been trained with n_features=%d" 
                             % (n_features, expected_n_features))
        #compute BM25 score
        bottom = X + self.k1*(1 - self.b + self._doc_len*(self.b/self._avgdl))
        weight = (X*(1+self.k1))/bottom
        score = weight * self._idf_diag
        
        #normalization
        if self.norm:
            score = normalize(score, norm=self.norm, copy=False)
        
        return score
        
    
    def fit_transform(self, X):
        """
        -	Train the BM25 model and return a vector-space representation of 
              the corpus
        -	Return a matrix where each row is one document, each column is 
             feature
        """
        self.fit(X)
        return self.transform(X)
        
        
    def get_vocabulary(self):
        """
        -	This method returns a list of unique words from the given corpus 
             during calling ‘fit’ method
        """
        return self.tf_vectorizer.get_feature_names()


if __name__ == '__main__':
    
    categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]

    remove = ('headers', 'footers', 'quotes')

    print("Loading 20 newsgroups dataset for categories:")
    print(categories if categories else "all")

    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=remove)
    
    data_test = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42,
                                   remove=remove)
    print('data loaded')

    # split a training set and a test set
    #y_train, y_test = data_train.target, data_test.target


    bm25 = BM25Vectorizer()
    score = bm25.fit_transform(data_train.data)
    print(score)





