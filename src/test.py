import tabu
import numpy as np
from k_medoids import KMedoids
from mean_shift import MeanShift


def test_points(layers=3, layer_distance=0.1, initial_side_length=0.5):
    points = []
    for layer in np.arange(layers):
        side_length = initial_side_length + layer * layer_distance
        points += [[side_length, -side_length]]
        points += [[-side_length, -side_length]]
        points += [[side_length, side_length]]
        points += [[-side_length, side_length]]
    return np.array(points)


if __name__ == '__main__':
    data_set = test_points()
    k_medoids = KMedoids(k=4, data_matrix=data_set)
    mean_shift = MeanShift(k=4, data_matrix=data_set)
    tabu = tabu.TabuSampler()
    samples = tabu.sample_qubo(mean_shift.matrix, timeout=5000)
    ground_truth_state = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    print(ground_truth_state @ mean_shift.matrix @ ground_truth_state)
    print(samples)
