import numpy as np
import scipy
from cvxopt import matrix, solvers
from tqdm import tqdm

def unmix_LS_unconstrained(M, abs):
    '''
    Unmixing algorithm using least squares with no constraints.
    input:
        M: endmember spectra matrix, np.array of shape (k, n) where k is the number of spectral bands and n is the number of endmembers
        abs: absorbance spectra, np.array of shape (...,k) where ... is the number of pixels
    output:
        c: estimated abundances, np.array of shape (...,n) where ... is the number of pixels
        err: difference spectrum, np.array of shape (...,k) where ... is the number of pixels
    '''
    M_inv = np.linalg.pinv(M)
    c = np.einsum("ij,...j->...i", M_inv, abs)
    err = np.einsum("kn,...n->...k", M, c) - abs
    return c, err

def unmix_LS_nonnegative(M, abs):
    '''
    Unmixing algorithm using least squares with non-negative constraints.
    input:
        M: endmember spectra matrix, np.array of shape (k, n) where k is the number of spectral bands and n is the number of endmembers
        abs: absorbance spectra, np.array of shape (k) or (m,l,k) where m and l are the number of pixels
    output:
        c: estimated abundances, np.array of shape (n) or (m,l,n)
        err: difference spectrum, np.array of shape (k) or (m,l,k)
    '''
    n = M.shape[1]
    if abs.ndim == 1:
        c, _ = scipy.optimize.nnls(M, abs)
    if abs.ndim == 3:
        m, l, k = abs.shape
        c = np.zeros((m,l,n))
        for i in tqdm(range(m)):
            for j in range(l):
                c[i,j,:], _ = scipy.optimize.nnls(M, abs[i,j,:])
    err = np.einsum("kn,...n->...k", M, c) - abs
    return c, err


class FCLSU:
    '''
    Fully Constrained Least Squares Unmixing (https://github.com/BehnoodRasti/Unmixing_Tutorial_IEEE_IADF/blob/main/VCA_FCLSU.ipynb)
    '''
    def __init__(self):
        pass

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    @staticmethod
    def _numpy_None_vstack(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.vstack([A1, A2])

    @staticmethod
    def _numpy_None_concatenate(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.concatenate([A1, A2])

    @staticmethod
    def _numpy_to_cvxopt_matrix(A):
        A = np.array(A, dtype=np.float64)
        if A.ndim == 1:
            return matrix(A, (A.shape[0], 1), "d")
        else:
            return matrix(A, A.shape, "d")

    def solve_FCLSU(self, Y, E):
        """
        Performs fully constrained least squares of each pixel in M
        using the endmember signatures of U. Fully constrained least squares
        is least squares with the abundance sum-to-one constraint (ASC) and the
        abundance nonnegative constraint (ANC).
        Parameters:
            Y: `numpy array`
                2D data matrix (L x N).
            E: `numpy array`
                2D matrix of endmembers (L x p).
        Returns:
            X: `numpy array`
                2D abundance maps (p x N).
        References:
            Daniel Heinz, Chein-I Chang, and Mark L.G. Fully Constrained
            Least-Squares Based Linear Unmixing. Althouse. IEEE. 1999.
        Notes:
            Three sources have been useful to build the algorithm:
                * The function hyperFclsMatlab, part of the Matlab Hyperspectral
                Toolbox of Isaac Gerg.
                * The Matlab (tm) help on lsqlin.
                * And the Python implementation of lsqlin by Valera Vishnevskiy, click:
                http://maggotroot.blogspot.ca/2013/11/constrained-linear-least-squares-in.html
                , it's great code.
        """
        assert len(Y.shape) == 2
        assert len(E.shape) == 2

        print(Y.shape)
        print(E.shape)

        L1, N = Y.shape
        L2, p = E.shape

        assert L1 == L2

        # Reshape to match implementation
        M = np.copy(Y.T)
        U = np.copy(E.T)

        solvers.options["show_progress"] = False

        U = U.astype(np.double)

        C = self._numpy_to_cvxopt_matrix(U.T)
        Q = C.T * C

        lb_A = -np.eye(p)
        lb = np.repeat(0, p)
        A = self._numpy_None_vstack(None, lb_A)
        b = self._numpy_None_concatenate(None, -lb)
        A = self._numpy_to_cvxopt_matrix(A)
        b = self._numpy_to_cvxopt_matrix(b)

        Aeq = self._numpy_to_cvxopt_matrix(np.ones((1, p)))
        beq = self._numpy_to_cvxopt_matrix(np.ones(1))

        M = np.array(M, dtype=np.float64)
        M = M.astype(np.double)
        X = np.zeros((N, p), dtype=np.float32)
        for n1 in tqdm(range(N)):
            d = matrix(M[n1], (L1, 1), "d")
            q = -d.T * C
            sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, None)["x"]
            X[n1] = np.array(sol).squeeze()
        return X.T

    def solve_FCLSU_2d(self, abs, M):
        '''
        Wrapper for solve_FCLSU to handle 2d images
        input: 
            abs: 3d array, hyperspectral image absorption where the last dimension is the spectral dimension
            M: 2d array of endmembers, where the first dimension is the spectral dimension
        output:
            c: 3d array, abundance maps of the endmembers
        '''
        m, l, k = abs.shape
        n = M.shape[1]
        Y = abs.reshape(-1, abs.shape[-1]).T  # (k, ml)
        c = self.solve_FCLSU(Y, M).T # (ml, n)
        c = c.reshape(m,l,n)  # (m, l, n)
        err = np.einsum("kn,mln->mlk", M, c) - abs
        return c, err