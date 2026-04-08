#
# TITLE: high performance computing with python numpy
# AUTHOR: Hyunseung Yoo
# PURPOSE: 
# REVISION: 
# REFERENCE: mastering numpy for data science and machine learning (M. J. Maxell, 2025)

import numpy as np
from typing import Optional, Tuple

#
# CH 14: Linear Regression from Scratch
#

class LinearRegressionGD:

    def __init__(self,
                 lr: float = 1e-2,
                 n_epochs: int = 1000,
                 batch_size: Optional[int] = None,  # None -> full-batch, 1 -> SGD, -1 -> mini-batch
                 alpha: float = 0.0,                # L2 regularization strength (Ridge)
                 fit_intercept: bool = True,
                 tol: float = 1e-6,
                 shuffle: bool = True,
                 verbose: bool = False,
                 rng: Optional[np.random.Generator] = None,):

        # input parameters
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.rng = rng or np.random.default_rng(0)

        #
        self.coef_ = None   # includes intercept if fit_intercept = True
        self.loss_history = []


    def _add_bias(self,
                  X: np.ndarray) -> np.ndarray:
        #
        if not self.fit_intercept:
            return X

        #
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([ones, X], axis=1)


    def _regularization_term(self,
                             w: np.ndarray) -> np.ndarray:
        # return vector to add to gradient for L2 penalty
        if self.alpha == 0.0:
            return 0.0
        
        if not self.fit_intercept:
            return (self.alpha / X.shape[0]) * w

        # do not regularize the intercept (first element)
        reg = (self.alpha / X.shape[0]) * w.copy()
        reg[0] = 0.0
        return reg


    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> 'LinearRegressionGD':

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape
        Xb = self._add_bias(X)  # shape (n, p+1) if intercept, else (n, p)
        m = Xb.shape[1]         # init weights (small random or zeros)

        self.coef_ = np.zeros(m, dtype=float)

        # set default batch_size
        if self.batch_size is None:
            batch_size = n      # full-batch
        else:
            batch_size = int(self.batch_size)

        #
        for epoch in range(self.n_epochs):
            #
            if self.shuffle:
                perm = self.rng.permutation(n)
                Xb = Xb[perm]
                y = y[perm]

            epoch_loss = 0.0

            for i in range(0, n, batch_size):
                xb = Xb[i:i+batch_size]
                yb = y[i:i+batch_size]
                pred = xb @ self.coef_      # (batch, )
                err = pred - yb             # (batch, )
                grad = (xb.T @ err) / xb.shape[0]   # (m,)

                # add L2 regularization (do not regularize intercept)
                if self.alpha != 0.0:
                    reg = (self.alpha / n) * self.coef_.copy()
                    if self.fit_intercept:
                        reg[0] = 0.0
                    grad += reg

                # gradient step
                self.coef_ = self.coef_ - self.lr * grad

                # accumulate loss (for monitoring)
                epoch_loss += 0.5 * (err**2).sum()

            epoch_loss / n
