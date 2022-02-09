import numpy as np

from qubo import QUBO
from kernel_functions import WelschKernel


class KMedoids(QUBO):
    """
    This class implements a QUBO formulation of the k-medoids problem. For the derivation see:
    Christian Bauckhage et al., "A QUBO Formulation of the k-Medoids Problem.”, LWDA, 2019.

    Attributes
    ----------
    n: int
        size of state space
    k: int
        number of clusters
    kernel_matrix: numpy matrix
        kernel matrix for representing the relations between the data points (default is Welsch's M-estimator)
    alpha, beta, gamma: float
        Lagrange parameters used for the QUBO optimization objective. The default values are suited for Welsch's
        M-estimator, they have to be adapted for other kernels.
    far_apart_matrix, central_matrix: numpy matrix
        QUBO matrices for the single problems contained in k-medoids, namely the identifaction of far apart data points
        and central data points.
    far_apart_offset, central_offset: numpy array
        corresponding offsets
    """

    def __init__(self, k: int, data_matrix=None, kernel=None, kernel_matrix=None, alpha=None, beta=None, gamma=2,
                 logging_dict=None):
        if kernel is None:
            kernel = WelschKernel()
        kernel_name = kernel.get_name()
        if logging_dict is None:
            logging_dict = {'qubo_name': 'k_medoids', 'kernel_matrix_name': 'kernel_matrix_' + kernel_name,
                            'qubo_matrix_name': 'qubo_matrix_{0}_k{1}_l{2}'.format(kernel_name, k, gamma),
                            'samples_name': 'samples_{0}_k{1}_l{2}'.format(kernel_name, k, gamma)}
        elif logging_dict['info'] is None:
            logging_dict['qubo_name'] = 'k_medoids'
            logging_dict['kernel_matrix_name'] = 'kernel_matrix_' + kernel_name
            logging_dict['qubo_matrix_name'] = 'qubo_matrix_{0}_k{1}_a{2}_b{3}_g{4}'.format(kernel_name, k, alpha, beta, gamma)
            logging_dict['samples_name'] = 'samples_{0}_k{1}_a{2}_b{3}_g{4}'.format(kernel_name, k, alpha, beta, gamma)
        else:
            logging_dict['qubo_name'] = 'k_medoids'
            logging_dict['kernel_matrix_name'] = 'kernel_matrix_' + kernel_name + '_' + logging_dict['info']
            logging_dict['qubo_matrix_name'] = 'qubo_matrix_{0}_k{1}_a{2}_b{3}_g{4}'.format(kernel_name, k, alpha,
                                                                                            beta, gamma) + '_' + logging_dict['info']
            logging_dict['samples_name'] = 'samples_{0}_k{1}_a{2}_b{3}_g{4}'.format(kernel_name, k, alpha, beta,
                                                                                    gamma) + '_' + logging_dict['info']

        super().__init__(None, logging_dict=logging_dict)
        self.k = k
        if self.matrix is None:
            # compute kernel matrix with given kernel if it is not given
            kernel_matrix = self._initialize_kernel_matrix(kernel, data_matrix, kernel_matrix)
            # normalization of matrix for lagrange parameters
            # print(self.kernel_matrix[:5, :5])
            maximum_entry = np.max(kernel_matrix)
            if maximum_entry > 1:
                kernel_matrix = kernel_matrix / maximum_entry
            # print(self.kernel_matrix[:5, :5])
            self.n = kernel_matrix.shape[0]
            assert self.n == kernel_matrix.shape[1]
            if gamma == 'k2':
                self.gamma = 1.0 / k**2
            elif gamma == 'k':
                self.gamma = 1.0 / k
            else:
                self.gamma = gamma
            if alpha == 'mean_shift' or beta == 'mean_shift':
                self.alpha = 1.0 / k**2
                self.gamma = self.gamma + self.alpha
                self.beta = 2 / (self.n * k)
            else:
                if alpha is None:
                    self.alpha = 1.0 / k
                else:
                    self.alpha = alpha
                if beta is None:
                    self.beta = 1.0 / self.n
                else:
                    self.beta = beta
            # print("k-Medoids parameters: alpha: {0}, beta: {1}, gamma: {2}".format(self.alpha, self.beta, self.gamma))
            self.far_apart_matrix, self.far_apart_offset = self._far_apart_objective(kernel_matrix)
            self.central_matrix, self.central_offset = self._central_objective(kernel_matrix)
            # compute the QUBO matrix Q and offset vector q for k-medoids
            matrix, offset = self._initialize_matrix_and_offset(kernel_matrix)
            self.matrix = self._compute_triangular(matrix, offset)

    def compute_far_apart_energy(self, state: np.ndarray):
        return self.compute_energy(state, self.far_apart_matrix, self.far_apart_offset)

    def compute_central_energy(self, state: np.ndarray):
        return self.compute_energy(state, self.central_matrix, self.central_offset)

    def _far_apart_objective(self, kernel_matrix):
        ones_matrix = np.ones(kernel_matrix.shape)
        ones_vector = np.ones(self.n)
        matrix = self.gamma * ones_matrix - self.alpha * 0.5 * kernel_matrix
        offset = -2 * self.gamma * self.k * ones_vector
        return matrix, offset

    def _central_objective(self, kernel_matrix):
        ones_vector = np.ones(self.n)
        ones_matrix = np.ones(kernel_matrix.shape)
        matrix = self.gamma * ones_matrix
        # beta is necessary here since o.w. the condition that exactly k medoids are created is not adhered
        offset = self.beta * kernel_matrix @ ones_vector - 2 * self.gamma * self.k * ones_vector
        return matrix, offset

    def _initialize_matrix_and_offset(self, kernel_matrix):
        ones_matrix = np.ones(kernel_matrix.shape)
        ones_vector = np.ones(self.n)
        # Identification of far apart data points
        matrix = self.gamma * ones_matrix - self.alpha * 0.5 * kernel_matrix
        # Identification of central data points
        offset = self.beta * kernel_matrix @ ones_vector - 2 * self.gamma * self.k * ones_vector
        return matrix, offset
