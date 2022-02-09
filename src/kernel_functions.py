import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist, squareform


class Kernel(ABC):
    """
    Abstract class for implementing general methods of a kernel, i.e. simple evaluation or the kernel matrix.
    """

    @abstractmethod
    def apply(self, x: np.ndarray, y: np.ndarray):
        """
        Abstract method for computing the kernel function k.

        :param x: first data point
        :param y: second data point
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def compute_kernel_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Method for computing the kernel matrix K. Data matrix X is given by X^T=[x_1,x_2,...,x_n], s.t. K is given by
        K_ij = k(x_i,x_j), where k is the kernel function.

        :param matrix: numpy array with single data points as rows
        """

        # pdist computes distance matrix in condensed form, may be more efficient to use that for computations
        kernel_matrix = squareform(pdist(matrix, metric=self.apply))
        return kernel_matrix


class WelschKernel(Kernel):
    def __init__(self, sigma=1):
        self.sigma = sigma

    def apply(self, x: np.ndarray, y: np.ndarray):
        return 1 - np.exp(-0.5 / self.sigma**2 * np.sum((x - y)**2))

    def get_name(self) -> str:
        return 'welsch'

    def compute_kernel_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return 1 - np.exp(- 0.5 / self.sigma * squareform(pdist(matrix, metric='sqeuclidean')))


class GaussianKernel(Kernel):
    def __init__(self, sigma=1):
        self.sigma = sigma

    def apply(self, x: np.ndarray, y: np.ndarray):
        return np.exp(-0.5 / self.sigma**2 * np.sum((x - y)**2))

    def get_name(self) -> str:
        return 'gaussian'

    def compute_kernel_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.exp(- 0.5 / self.sigma * squareform(pdist(matrix, metric='sqeuclidean')))


class EuclideanKernel(Kernel):
    def apply(self, x: np.ndarray, y: np.ndarray):
        return np.linalg.norm(x - y)

    def get_name(self) -> str:
        return 'euclidean'

    def compute_kernel_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return squareform(pdist(matrix, metric='euclidean'))


class ManhattenKernel(Kernel):
    def apply(self, x: np.ndarray, y: np.ndarray):
        return np.sum(np.abs(x - y))

    def get_name(self) -> str:
        return 'manhatten'

    def compute_kernel_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return squareform(pdist(matrix, metric='cityblock'))
