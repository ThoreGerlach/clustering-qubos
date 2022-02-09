import tabu
import numpy as np
from src.k_medoids import KMedoids


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
    tabu = tabu.TabuSampler()
    matrix = np.ones((10, 10))
    samples = tabu.sample_qubo(k_medoids.matrix)
    print(samples)