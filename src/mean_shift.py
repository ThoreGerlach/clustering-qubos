import numpy as np
from qubo import QUBO
from kernel_functions import GaussianKernel


class MeanShift(QUBO):
    """
    This class implements a QUBO formulation of the mean shift problem. For the derivation see:
    Bauckhage, Christian, Fabrice Beaumont, and Sebastian Müller. "ML2R Coding Nuggets Hopfield Nets for Hard Vector
    Quantization."

    Attributes
    ----------
    n: int
        size of state space
    k: int
        number of clusters
    kernel_matrix: numpy matrix
        kernel matrix for representing the relations between the data points (default is standard Gaussian kernel)
    alpha: float
        Lagrange parameter used for the QUBO optimization objective (default is 1)
    """

    def __init__(self, k: int, data_matrix=None, kernel=None, kernel_matrix=None, alpha=1, logging_dict=None):
        if kernel is None:
            kernel = GaussianKernel()
        kernel_name = kernel.get_name()
        if logging_dict is None:
            logging_dict = {'qubo_name': 'mean_shift', 'kernel_matrix_name': 'kernel_matrix_' + kernel_name,
                            'qubo_matrix_name': 'qubo_matrix_{0}_k{1}_l{2}'.format(kernel_name, k, alpha),
                            'samples_name': 'samples_{0}_k{1}_l{2}'.format(kernel_name, k, alpha)}
        elif logging_dict['info'] is None:
            logging_dict['qubo_name'] = 'mean_shift'
            logging_dict['kernel_matrix_name'] = 'kernel_matrix_' + kernel_name
            logging_dict['qubo_matrix_name'] = 'qubo_matrix_{0}_k{1}_l{2}'.format(kernel_name, k, alpha)
            logging_dict['samples_name'] = 'samples_{0}_k{1}_l{2}'.format(kernel_name, k, alpha)
        else:
            logging_dict['qubo_name'] = 'mean_shift'
            logging_dict['kernel_matrix_name'] = 'kernel_matrix_' + kernel_name + '_' + logging_dict['info']
            logging_dict['qubo_matrix_name'] = 'qubo_matrix_{0}_k{1}_l{2}'.format(kernel_name, k, alpha) + '_' + \
                                               logging_dict['info']
            logging_dict['samples_name'] = 'samples_{0}_k{1}_l{2}'.format(kernel_name, k, alpha) + '_' + \
                                           logging_dict['info']
        super().__init__(None, logging_dict=logging_dict)
        self.k = k
        if self.matrix is None:
            # compute kernel matrix with given kernel if it is not given
            kernel_matrix = self._initialize_kernel_matrix(kernel, data_matrix, kernel_matrix)
            maximum_entry = np.max(kernel_matrix)
            if maximum_entry > 1:
                kernel_matrix = kernel_matrix / maximum_entry
            self.n = kernel_matrix.shape[0]
            assert self.n == kernel_matrix.shape[1]
            self.alpha = alpha
            # compute the QUBO matrix Q and offset vector q for mean shift
            matrix, offset = self._initialize_matrix_and_offset(kernel_matrix)
            self.matrix = self._compute_triangular(matrix, offset)

    def _initialize_matrix_and_offset(self, kernel_matrix):
        ones_matrix = np.ones(kernel_matrix.shape)
        ones_vector = np.ones(self.n)
        # Stems from scalar product of Parzen density estimates of cluster centers with itself, (phi_Y, phi_Y)
        matrix = (1.0 / self.k**2) * kernel_matrix + self.alpha * ones_matrix
        # Stems from scalar product of Parzen estimates of cluster centers with Parzen estimates of data, (phi_Y, phi_X)
        offset = - 2.0 / (self.n * self.k) * (kernel_matrix @ ones_vector) - 2 * self.alpha * self.k * ones_vector
        return matrix, offset
