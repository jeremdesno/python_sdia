import numpy as np
cimport numpy as cnp
import cython

@cython.boundscheck(False)
@cython.wraparound(False)

def knn_cython(cnp.ndarray[double, ndim=2] train_df,cnp.ndarray[double, ndim=2] test_df, int n_neighbours):
    cdef Py_ssize_t num_samples_train = train_df.shape[0]
    cdef Py_ssize_t num_samples_test = test_df.shape[0]
    cdef int i, j
    cdef cnp.ndarray[double, ndim=1] distances  
    cdef cnp.ndarray[Py_ssize_t, ndim=1] sorted_indices
    cdef cnp.ndarray[Py_ssize_t, ndim=1] nearest_neighbors
    cdef cnp.ndarray[double, ndim=1] neighbor_labels
    cdef int label
    predictions = np.zeros(num_samples_test, dtype = int)

    # Create memoryviews for accessing the data
    cdef double[:, :] train_data = train_df
    cdef double[:, :] test_data = test_df
    cdef double distance


    for i in range(num_samples_test):
        distances = np.empty(num_samples_train, dtype=np.double)
        for j in range(num_samples_train):
            distance = 0.0
            for k in range(train_data.shape[1]):
                if k != 1:
                    distance += (train_data[j][k] - test_data[i][k]) ** 2
            distances[j] = distance

        sorted_indices = np.argsort(distances)
        nearest_neighbors = sorted_indices[:n_neighbours]

        neighbor_labels = train_df[nearest_neighbors, 1]  
        label = np.argmax(np.bincount(neighbor_labels.astype(int)))

        predictions[i] = label

    return np.array(predictions)
