import torch
import time
from tqdm import tqdm

class KMEANS:
    def __init__(self, n_clusters=16, max_iter=1000, verbose=False,device = torch.device("cuda")):

        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device
        self.alpha = 0.8
        self.stable_centers = None

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        if self.stable_centers is None:
            init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
            init_points = x[init_row]
        else:
            init_points = self.stable_centers
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break
            self.count += 1
        
        if (self.centers != self.centers).sum() != 0:
            self.centers = init_points
            return
        
        if self.stable_centers is None:
            self.stable_centers = self.centers
        else:
            self.stable_centers = self.alpha * self.stable_centers + (1-self.alpha) * self.centers

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def find_closest_centers(self, x):
        if self.stable_centers is None:
            return None
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        closest_centers = torch.empty((x.shape[0], self.centers.shape[1])).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.stable_centers, sample - self.stable_centers), (1))
            labels[i] = torch.argmin(dist)
        return labels
    

def time_clock(matrix,device):
    a = time.time()
    k = KMEANS(max_iter=10,verbose=False,device=device)
    k.fit(matrix)
    b = time.time()
    return (b-a)/k.count

def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()

    device = choose_device(False)

    cpu_speeds = []
    for i in tqdm([20,100,500]):
        matrix = torch.rand((10000,i)).to(device)
        speed = time_clock(matrix,device)
        cpu_speeds.append(speed)
    l1, = plt.plot([20,100,500],cpu_speeds,color = 'r',label = 'CPU')

    device = choose_device(True)

    gpu_speeds = []
    for i in tqdm([20, 100, 500, 2000, 8000, 20000]):
        matrix = torch.rand((10000, i)).to(device)
        speed = time_clock(matrix,device)
        gpu_speeds.append(speed)
    l2, = plt.plot([20, 100, 500, 2000, 8000, 20000], gpu_speeds, color='g',label = "GPU")



    plt.xlabel("num_features")
    plt.ylabel("speed(s/iter)")
    plt.title("Speed with cuda")
    plt.legend(handles = [l1,l2],labels = ['CPU','GPU'],loc='best')
    plt.savefig(".speed.jpg")