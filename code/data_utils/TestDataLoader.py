import numpy as np
import warnings
import os
# import data_utils.augmentation as augmentation
# import augmentation as augmentation
from torch.utils.data import Dataset
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

class FAFBDataset(Dataset):
    def __init__(self, root, npoint=1024, uniform=False, normal_channel=True, cache_size=15000, valdataset = '', catfile = 'classname.txt', istrain=True):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, catfile)
        self.normal_channel = normal_channel

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_names = {}
        for i in self.cat:
            shape_names[i] = [line.rstrip() for line in open(os.path.join(self.root, '{}.txt'.format(i)))]
            
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(cat, os.path.join(self.root, cat, shape_names[cat][i]))
                          for cat in shape_names.keys() for i in range(len(shape_names[cat]))]
        print('The size of all data is %d'%(len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        fn = self.datapath[index]
        name =  list(self.datapath[index])[1].split('/')[-1].split('_')[-1].split('.')[0]
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints,:]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls)

        return point_set, cls, name
        
#         if self.istrain:
#             point_view1 = augmentation.transform(point_set)
#             point_view2 = augmentation.transform(point_set)
#             if self.uniform:
#                 point_view1 = farthest_point_sample(point_view1, self.npoints)
#                 point_view2 = farthest_point_sample(point_view2, self.npoints)
#             else:
#                 point_view1 = point_view1[0:self.npoints,:]
#                 point_view2 = point_view2[0:self.npoints,:]

#             point_view1[:, 0:3] = pc_normalize(point_view1[:, 0:3])
#             point_view2[:, 0:3] = pc_normalize(point_view2[:, 0:3])


#             if not self.normal_channel:
#                 point_view1 = point_view1[:, 0:3]
#                 point_view2 = point_view2[:, 0:3]

#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls)
                
#             return (point_view1, point_view2), cls
#         else:
#             if self.uniform:
#                 point_set = farthest_point_sample(point_set, self.npoints)
#             else:
#                 point_set = point_set[0:self.npoints,:]

#             point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

#             if not self.normal_channel:
#                 point_set = point_set[:, 0:3]

#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls)

#             return point_set, cls, name
#, fn[1].split('/')[-1].split('_')[-1].split('.')[0]
    def __getitem__(self, index):
        return self._get_item(index)

def visualizePointCloud(points, savePath):
    """
    Input:
        points: pointcloud data, [N, D]
    """
    x = [k[0] for k in points]
    y = [k[1] for k in points]
    z = [k[2] for k in points]
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Points cloud')
    ax.scatter(x, y, z, c='b', marker='.', s=10, linewidth=0, alpha=1, cmap='spectral')
    plt.savefig(savePath)

if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import augmentation as augmentation
    data = FAFBDataset('/data/chih0321/FAFB_Subcompartment', uniform=True, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label, name in DataLoader:
        point = point.numpy()
#         for i in range(len(point)):
#             visualizePointCloud(point[i], 'test'+str(i)+'.png')
#             np.savetxt('test'+str(i)+'.txt', point[i])
        print(label)
        #print(point)
        print(name)