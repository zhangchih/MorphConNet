import numpy as np
import random

def rotation_3d_in_axis(points, angles, axis=0):
    if len(points.shape) == 2:
        points = np.expand_dims(points, axis=0)
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def get_partitions(points):
    # partition num = 8
    partition1 = points[(points[..., 0] > 0)*(points[..., 1] > 0)*(points[..., 2] > 0)]
    partition2 = points[(points[..., 0] < 0)*(points[..., 1] > 0)*(points[..., 2] > 0)]
    partition3 = points[(points[..., 0] < 0)*(points[..., 1] < 0)*(points[..., 2] > 0)]
    partition4 = points[(points[..., 0] > 0)*(points[..., 1] < 0)*(points[..., 2] > 0)]
    partition5 = points[(points[..., 0] > 0)*(points[..., 1] > 0)*(points[..., 2] < 0)]
    partition6 = points[(points[..., 0] < 0)*(points[..., 1] > 0)*(points[..., 2] < 0)]
    partition7 = points[(points[..., 0] < 0)*(points[..., 1] < 0)*(points[..., 2] < 0)]
    partition8 = points[(points[..., 0] > 0)*(points[..., 1] < 0)*(points[..., 2] < 0)]
    
    return [partition1, partition2, partition3, partition4, partition5, partition6, partition7, partition8]

def dropout_partition(partitions):
    i = random.randint(0, len(partitions)-1)
    del partitions[i]
    return partitions

def merge_partitions(partitions):
    return np.concatenate(partitions, axis=0)

def sparsify_partition(partitions):
    i = random.randint(0, len(partitions)-1)
    partition = partitions[i].copy()
    del partitions[i]
    if partition.shape[0] != 0:
        partition = farthest_point_sample(partition, int(partition.shape[0]*2/3))
    partitions.append(partition)
    return partitions

def add_noisy(points, noisy_nums=50):
    noisy_points = np.random.rand(noisy_nums, 3).astype(np.float32)
    points = np.concatenate((points, noisy_points), 0)
    return points

def random_rotation(points, dev=10):
    rotation_3d_in_axis(points, [random.uniform(-1, 1)*dev], axis=0)
    rotation_3d_in_axis(points, [random.uniform(-1, 1)*dev], axis=1)
    rotation_3d_in_axis(points, [random.uniform(-1, 1)*dev], axis=2)
        
    
def transform(points):
    """
        intput:
            points: x, y, z of points, [N, 3]
        return:
            augmentation_points: x, y, z of points, [M, 3]
    """
    partitions = get_partitions(points)
#     # 以概率0.4随机丢失partition
#     if random.random() < 0.4:
#         partitions = dropout_partition(partitions)
    # 以概率0.6随机Sparsify partition
    if random.random() < 0.6:
        partitions = sparsify_partition(partitions)
    augmentation_points = merge_partitions(partitions)
    # 以概率0.5加入噪声点
    if random.random() < 0.5:
        augmentation_points = add_noisy(augmentation_points)
    # 以概率0.5旋转旋转
#     if random.random() < 0.5:
#         augmentation_points = random_rotation(augmentation_points)
    return augmentation_points