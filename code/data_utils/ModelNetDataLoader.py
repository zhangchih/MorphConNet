import numpy as np
import warnings
import os
import data_utils.augmentation as augmentation
from torch.utils.data import Dataset
import h5py
warnings.filterwarnings('ignore')



def pc_normalize(pc):
    #print('-----------------pc',pc.shape)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    #print('-----------------pc_cen', pc.shape)
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    #print('-----------------pc_nor', pc.shape)
    return pc

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

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, valdataset = '', istrain=True):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
#         self.catfile = os.path.join(self.root, 'neuron_dataset_shape_names.txt')
        self.valdataset = valdataset
        self.istrain = istrain

#         self.cat = [line.rstrip() for line in open(self.catfile)]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel
#         print('-------------------', self.normal_channel)
        data_dict = h5py.File(root, 'r')
        self.data_points = data_dict['pc']
        self.label = data_dict['label']
#         shape_ids = {}
# #         shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, self.valdataset, 'neuron_dataset_train.txt'))]
# #         遗留问题，数据集为师兄创建不可改，因此在全部数据集训练时，需要将训练集的文件改为测试集的名字，正常训练使用上一行即可
#         shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, self.valdataset, 'neuron_dataset_test.txt'))] # 即train数据集需要从neuron_dataset_test.txt中读取文件
#         shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, self.valdataset, 'neuron_dataset_test.txt'))]

#         assert (split == 'train' or split == 'test')
#         shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        
#         # list of (shape_name, shape_txt_file_path) tuple
#         self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
#                          in range(len(shape_ids[split]))]
#         print('The size of %s data is %d'%(split,len(self.datapath)))

#         self.cache_size = cache_size  # how many data points to cache in memory
#         self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return self.data_points.shape[0]

    def _get_item(self, index):
#         fn = self.datapath[index]
#         name =  self.datapath[index][1].split('/')[-1].split('_')[-1].split('.')[0]
#         if index in self.cache:
#             point_set, cls = self.cache[index]
#         else:
#             cls = self.classes[self.datapath[index][0]]
#             cls = np.array([cls]).astype(np.int32)
#             point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        point_set = self.data_points[index]
        N, D = point_set.shape
        if N <= 0:
            point_set = self.data_points[0]
        cls = self.label[index].astype(np.int32)
        
        if self.istrain:
            point_view1 = augmentation.transform(point_set)
            point_view2 = augmentation.transform(point_set)
            N, D = point_view1.shape
            while N <= 0:
                point_view1 = augmentation.transform(point_set)
                N, D = point_view1.shape
            N, D = point_view2.shape
            while N <= 0:
                point_view2 = augmentation.transform(point_set)
                N, D = point_view2.shape
                
            if self.uniform:
                point_view1 = farthest_point_sample(point_view1, self.npoints)
                point_view2 = farthest_point_sample(point_view2, self.npoints)
            else:
                point_view1 = point_view1[0:self.npoints,:]
                point_view2 = point_view2[0:self.npoints,:]

            point_view1[:, 0:3] = pc_normalize(point_view1[:, 0:3])
            point_view2[:, 0:3] = pc_normalize(point_view2[:, 0:3])


            if not self.normal_channel:
                point_view1 = point_view1[:, 0:3]
                point_view2 = point_view2[:, 0:3]

#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls)
                
            return (point_view1, point_view2), cls
        else:
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls)

            return point_set, cls
#, fn[1].split('/')[-1].split('_')[-1].split('.')[0]
    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import augmentation as augmentation
    data = ModelNetDataLoader('/data/nerdxie/neuron_cluster/neuron_dataset_1',split='test', uniform=True, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=False)
    for (point_view1, point_view2), label in DataLoader:
        point = point_view1.numpy()
        for i in range(len(point)):
            visualizePointCloud(point[i], 'test'+str(i)+'.png')
            np.savetxt('test'+str(i)+'.txt', point[i])
        print(point.shape)
        #print(point)
        print(label.shape)