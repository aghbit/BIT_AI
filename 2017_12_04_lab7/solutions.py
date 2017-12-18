import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(int(time.time()))


def generate_mess(clusters_num=3, closeness=None, point_range=100, points_in_cluster=50):
    closeness = 2 / clusters_num if closeness is None else closeness
    clusters = np.random.rand(clusters_num, 2) * point_range
    points = ((np.random.rand(points_in_cluster, clusters_num, 2) * point_range - (
        point_range / 2)) * closeness + clusters).reshape(-1, 2)

    return points


def k_means_iteration(centroids, datapoints):
    c_t = np.array([c.reshape(-1, 1) for c in centroids.T])
    d_t = np.array([d.reshape(-1, 1) for d in datapoints.T])

    diffs = np.array([
        (c_t[i].reshape(-1) ** 2 - 2 * (d_t[i] @ c_t[i].T)).T + d_t[i].reshape(-1) ** 2
        for i in range(len(c_t))])

    diffs = diffs.sum(axis=0)

    cluster_assignments = np.argmin(diffs, axis=0)
    clusters = np.array([np.argwhere(cluster_assignments == i) for i in range(len(centroids))])
    c_t = np.array(
        [
            [
                d_t[i][c].mean() if d_t[i][c].size != 0 else d_t[i].mean() for c in clusters
            ]
            for i in range(len(c_t))
        ])

    return c_t.T, clusters

#
# def demonstrate_k_means(k_means_iter, num_iterations, centroids, datapoints):
#     _, clusters = k_means_iter(centroids, datapoints)
#
#     centroids_diffs = np.zeros(centroids.shape)
#     for i in range(num_iterations):
#         prev_centroids = centroids
#         centroids, clusters = k_means_iter(centroids, datapoints)
#         centroids_diffs = centroids - prev_centroids
#
#     for c in clusters:
#         plt.scatter(datapoints[:, 0][c], datapoints[:, 1][c])
#
#     plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='.')
#     print(centroids_diffs)
#     plt.show()
#
#
# def k_means_img_compression(k_means_iter, num_iterations, centroids, image):
#     datapoints = np.copy(image.reshape(-1, 3))
#     _, clusters = k_means_iter(centroids, datapoints)
#     for i in range(num_iterations):
#         centroids, clusters = k_means_iter(centroids, datapoints)
#
#     for i in range(len(clusters)):
#         datapoints[clusters[i]] = centroids[i]
#     img_reconstructed = datapoints.reshape(image.shape)
#     print(num_iterations)
#     plt.imshow(image[:, :, [2, 1, 0]])
#     plt.show()
#     plt.imshow(img_reconstructed[:, :, [2, 1, 0]])
#     plt.show()
