import os
import warnings
import numpy as np
import helper as helper
from kernel_functions import Kernel


default_qubo_logging_dict = {'log_folder': '.', 'qubo_name': 'qubo', 'qubo_matrix_name': 'qubo_matrix',
                             'kernel_matrix_name': 'kernel_matrix', 'samples_name': 'samples',
                             'info': None, 'logging': False}


class QUBO:
    """
    This is a class implements a general interface for a quadratic unconstrained binary optimization (QUBO) problem.

    It is possible to choose between two binary representations, namely boolean and bipolar. Binary boolean variables
    z_i can take values in {0,1} while bipolar variables s_i can take values in {-1,1}. We assume the boolean
    standard form, i.e. the objective value of a given n-dim boolean vector z is given by: z^T Q z, where Q is an upper
    triangular matrix. For bipolar vectors s we use the form s^T Q' s + q'^T s, where Q' is a symmetric matrix.

    Attributes
    ----------
    matrix: np.ndarray
        nxn numpy matrix representing Q/Q'
    offset: np.ndarray, optional
        n-dim numpy vector representing q' (default is None for boolean version)
    bipolar: boolean, optional
        variable indicating whether the problem is boolean or bipolar (default is False)
    n: int
        size of state space
    logging: bool, optional
        whether to log QUBO data (default is False)
    logging_dict: dict, optional
        dictionary for logging purposes (default is default_qubo_logging_dict)
    """

    def __init__(self, matrix, bipolar=False, offset=None, logging_dict=None):
        self.logging_dict = helper.fill_dict(logging_dict, default_qubo_logging_dict)
        try:
            self.n = int(matrix.shape[0])
            assert self.n == matrix.shape[1]
            if offset is not None:
                assert self.n == offset.shape[0]

            if not bipolar:
                matrix = self._compute_triangular(matrix=matrix, offset=offset)
                self.offset = None
            else:
                if not np.allclose(matrix, matrix.T):
                    raise Exception('Bipolar QUBO matrix has to be symmetric!')
                self.offset = offset
        except Exception:
            pass
        self.matrix = matrix
        self.bipolar = bipolar

    def _compute_triangular(self, matrix, offset):
        """
        Method for computing an upper triangular matrix of the underlying QUBO problem. If it is precomputed it is
        simply loaded and computed and saved otherwise.

        :param matrix: Instance of class QUBO
        :param offset: If qubo is a numpy matrix, the offset can be given by a numpy array
        """

        if self.logging_dict['logging']:
            qubo_matrices_folder = os.path.join(self.logging_dict['log_folder'], self.logging_dict['qubo_name'],
                                                'qubo_matrices')
            file_path = os.path.join(qubo_matrices_folder, self.logging_dict['qubo_matrix_name'] + '.npy')
            if not os.path.exists(qubo_matrices_folder):
                os.makedirs(qubo_matrices_folder)
            if os.path.isfile(file_path):
                # start = time.time()
                # print('Loading ' + file_path)
                triangular_qubo = np.load(file_path)
                # end = time.time()
                # print('Loaded in {} s!'.format(end - start))
            else:
                # start = time.time()
                # print('Computing and saving ' + file_path)
                triangular_qubo = QUBO.to_upper_triangular(matrix=matrix, offset=offset)
                np.save(file_path, triangular_qubo)
                # end = time.time()
                # print('Saved in {} s!'.format(end - start))
        else:
            triangular_qubo = QUBO.to_upper_triangular(matrix=matrix, offset=offset)
        return triangular_qubo

    @staticmethod
    def to_upper_triangular(matrix, offset=None):
        """
        This method returns a single upper triangular matrix Q_u, s.t.
        argmin_z z^T Q z + q^T z = argmin_z z^T Q_u z.
        If this is not possible, an exception is raised. Only works if QUBO is already in boolean version.
        """

        if offset is not None:
            np.fill_diagonal(matrix, matrix.diagonal() + offset)
        if np.allclose(np.triu(matrix), matrix):
            # print('Already triangular!')
            return matrix
        elif np.allclose(np.tril(matrix), matrix):
            return matrix.T
        elif np.allclose(matrix, matrix.T):
            # print('Symmetric!')
            return QUBO.make_triangular(matrix)
        else:
            raise Exception("It is not possible to form the matrix to upper triangular!")
        pass

    def to_boolean(self):
        """
        This method converts Q'->Q, q'->None using the transformation s -> (2 * z - 1), s.t.
        argmin_s s^T Q' s + q'^T s = argmin_z z^T Q z.
        """

        if self.bipolar:
            row_sums = np.sum(self.matrix, axis=1)
            column_sums = np.sum(self.matrix, axis=0)
            self.matrix = 4 * self.matrix.copy()
            offset = 2 * (self.offset - row_sums - column_sums)
            np.fill_diagonal(self.matrix, self.matrix.diagonal() + offset)
            self.make_triangular(self.matrix)
            self.offset = None
            self.bipolar = False
        else:
            print('QUBO is already in boolean form!')
        pass

    def to_bipolar(self):
        """
        This method converts Q->Q', q->q' using the transformation z -> (s + 1) / 2, s.t.
        argmin_s s^T Q' s + q'^T s = argmin_z z^T Q z + q^T z.
        """

        if not self.bipolar:
            row_sums = np.sum(self.matrix, axis=1)
            column_sums = np.sum(self.matrix, axis=0)
            self.matrix = 0.25 * self.matrix
            self.offset = 0.25 * (row_sums + column_sums)

            self.bipolar = True
        else:
            print('QUBO is already in bipolar form!')
        pass

    def energy_residual(self) -> float:
        """
        Returns the residual energy which stems from reformulating QUBO between bipolar and boolean, i.e.
        s^T Q' s + q'^T s = z^T Q z + q^T z + C, where C is a constant residual.
        """
        matrix_sum = np.sum(self.matrix)
        if self.bipolar:
            offset_sum = np.sum(self.offset)
            residual = matrix_sum - offset_sum
        else:
            # residual = 0.25 * matrix_sum + 0.5 * offset_sum
            residual = 0.25 * matrix_sum
        return residual

    def compute_energy(self, state: np.ndarray, matrix=None, offset=None) -> float:
        """
        This method returns the QUBO energy of a given state z/s, i.e. z^T Q z + q^T z / s^T Q' s + q'^T s.

        :param state: State for energy evaluation. Either boolean or bipolar.
        :param matrix: For "static" use of method
        :param offset: For "static" use of method
        """

        state_boolean = self._is_boolean(state)
        state_bipolar = self._is_bipolar(state)
        if not state_boolean and not state_bipolar:
            raise Exception('State is neither boolean nor bipolar!')
        if self.bipolar:
            if state_boolean and not state_bipolar:
                raise Exception('QUBO is bipolar, state is boolean!')
        else:
            if state_bipolar and not state_boolean:
                raise Exception('QUBO is boolean, state is bipolar!')
        if matrix is None:
            if self.offset is None:
                energy = state.T @ self.matrix @ state
            else:
                energy = state.T @ self.matrix @ state + self.offset @ state
        else:
            if offset is None:
                energy = state.T @ matrix @ state
            else:
                energy = state.T @ matrix @ state + offset @ state
        return energy

    def compute_energies(self, states: np.ndarray, matrix=None, offset=None):
        energies = np.zeros(states.shape[0])
        for i in np.arange(len(energies)):
            energies[i] = self.compute_energy(states[i], matrix=matrix, offset=offset)
        return energies

    @staticmethod
    def bipolar_to_boolean(state: np.ndarray) -> np.ndarray:
        """
        This method converts a bipolar state s to a boolean state z via z = (s + 1) / 2

        :param state: Bipolar state for conversion to boolean.
        """
        state_boolean = QUBO._is_boolean(state)
        state_bipolar = QUBO._is_bipolar(state)
        if state_boolean:
            print('State is already boolean!')
        elif not state_bipolar:
            raise Exception('State is neither boolean nor bipolar!')
        else:
            return (state + 1) / 2

    @staticmethod
    def boolean_to_bipolar(state: np.ndarray) -> np.ndarray:
        """
        This method converts a boolean state z to a bipolar state s via s = 2 * z - 1

        :param state: Boolean state for conversion to bipolar.
        """

        state_boolean = QUBO._is_boolean(state)
        state_bipolar = QUBO._is_bipolar(state)
        if state_bipolar:
            print('State is already bipolar!')
            return state
        elif not state_boolean:
            raise Exception('State is neither boolean nor bipolar!')
        else:
            return 2 * state - 1

    def _get_split_indices(self, split_method, split_size):
        if split_method == 'uniform':
            rng = np.random.default_rng()
            split_indices = np.sort(rng.choice(self.n, split_size, replace=False))
        else:
            raise Exception('Unknown splitting method!')
        return split_indices

    def split(self, split_size: int, initial_state: np.ndarray, split_indices, residual=False):
        """
        Splits a QUBO problem into a smaller sub-QUBO problem, with fixing indices with given values.

        :param split_size: Size of the sub-QUBO
        :param initial_state: Initial state for fixing values for cutset indices
        :param split_indices: Indices of sub-QUBO, which are optimized (optional). If they are not given, the indices
            are chosen uniformly random
        :param residual: Boolean value whether to compute the residual. Very time inefficient for large QUBOs
        """

        mask = np.ones(self.n, dtype=bool)
        mask[split_indices] = False
        assert len(split_indices) == split_size
        cutset_indices = np.arange(self.n)[mask]
        return self._compute_split(initial_state, split_indices, cutset_indices, residual)

    def _compute_split(self, initial_state: np.ndarray, split_indices: np.ndarray, cutset_indices: np.ndarray,
                       residual: bool):
        split_matrix = self.matrix[split_indices][:, split_indices]
        cutset_split_matrix_row = self.matrix[split_indices][:, cutset_indices]
        cutset_split_matrix_column = self.matrix[cutset_indices][:, split_indices]
        cutset_initial_state = initial_state[cutset_indices]
        row_cutset = cutset_split_matrix_row @ cutset_initial_state
        column_cutset = cutset_initial_state @ cutset_split_matrix_column
        offset = row_cutset + column_cutset
        if residual:
            cutset_matrix = self.matrix[cutset_indices][:, cutset_indices]
            split_residual = cutset_initial_state @ cutset_matrix @ cutset_initial_state
        else:
            split_residual = None
        if not self.bipolar:
            split_qubo = QUBO(matrix=split_matrix, offset=offset)
        else:
            offset = offset + self.offset
            split_qubo = QUBO(matrix=split_matrix, offset=offset, bipolar=True)
            if residual:
                split_residual = split_residual + np.sum(self.offset[cutset_indices])
        return split_qubo, split_indices, split_residual

    def _initialize_kernel_matrix(self, kernel, data_matrix, kernel_matrix):
        if kernel_matrix is None:
            if data_matrix is None:
                raise Exception('Kernel matrix is not computable without data!')
            else:
                if self.logging_dict['logging']:
                    kernel_matrix_folder = os.path.join(self.logging_dict['log_folder'], self.logging_dict['qubo_name'],
                                                        'kernel_matrices')
                    file_path = os.path.join(kernel_matrix_folder, self.logging_dict['kernel_matrix_name'] + '.npy')
                    if not os.path.exists(kernel_matrix_folder):
                        os.makedirs(kernel_matrix_folder)
                    if os.path.isfile(file_path):
                        # start = time.time()
                        # print('Loading kernel matrix: ' + file_path)
                        kernel_matrix = np.load(file_path)
                        # end = time.time()
                        # print('Loaded kernel matrix in {} s!'.format(end - start))
                    else:
                        assert isinstance(kernel, Kernel)
                        # start = time.time()
                        # print("Computing and saving kernel matrix: ")
                        kernel_matrix = kernel.compute_kernel_matrix(data_matrix)
                        np.save(file_path, kernel_matrix)
                        # end = time.time()
                        # print("Saved kernel matrix in {} s!".format(end - start))
                else:
                    kernel_matrix = kernel.compute_kernel_matrix(data_matrix)
        return kernel_matrix

    @staticmethod
    def _is_boolean(state: np.ndarray) -> bool:
        for entry in state:
            if np.abs(entry) > 1e-8 and np.abs(entry - 1) > 1e-8:
                return False
        return True

    @staticmethod
    def _is_bipolar(state: np.ndarray) -> bool:
        for entry in state:
            if np.abs(entry + 1) > 1e-8 and np.abs(entry - 1) > 1e-8:
                return False
        return True

    @staticmethod
    def make_triangular(matrix, lower=False) -> np.ndarray:
        assert matrix.shape[0] == matrix.shape[1]
        triangular_matrix = np.triu(matrix + np.tril(matrix, -1).T)
        if lower:
            triangular_matrix = triangular_matrix.T
        return triangular_matrix
