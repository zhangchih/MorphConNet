import torch
import sys
import yaml
from torchvision import transforms, datasets
import torchvision
import numpy as np
import os
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader

sys.path.append('../')
from models.pointnetv2_encoder import PointNetV2

batch_size = 40

config = yaml.load(open("../config/config.yaml", "r"), Loader=yaml.FullLoader)
# from data_utils.ModelNetDataLoader import ModelNetDataLoader
# train_dataset = ModelNetDataLoader('/data/nerdxie/neuron_cluster/neuron_dataset_1',split='train', uniform=True, normal_channel=True,istrain=False,)
# test_dataset = ModelNetDataLoader('/data/nerdxie/neuron_cluster/neuron_dataset_1',split='test', uniform=True, normal_channel=True,istrain=False,)
from data_utils.FAFBAnalysisDataLoader import TestDataLoader
# full_dataset = ModelNetDataLoader('./data_neuron.h5', uniform=True, normal_channel=True, istrain=False)
test_dataset = TestDataLoader('/data/nerdxie/neuron_cluster/neuron_dataset_add_all_128/',split='train', uniform=True, normal_channel=True,istrain=False,)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                          num_workers=4, drop_last=True, shuffle=True)
device = 'cuda' #'cuda' if torch.cuda.is_available() else 'cpu'
encoder = PointNetV2(**config['network']).to(device)
# encoder.projetion = torch.nn.Sequential()
encoder = torch.nn.DataParallel(encoder).cuda()
#load pre-trained parameters

# for i in load_params['online_network_state_dict'].keys():
#     if 'projetion' in i:
#         del load_params['online_network_state_dict'][i]

encoder.load_state_dict(torch.load(os.path.join('encoder1.pt'),
                         map_location=torch.device(torch.device(device))))
print("Parameters successfully loaded.")

# remove the projection head
# encoder = torch.nn.Sequential(*list(encoder.cpu().children())[:-1])    
# encoder = encoder.to(device)
# output_feature_dim = 1024
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)
device = 'cuda'
logreg = LogisticRegression(1024, 3)
logreg = logreg.to(device)
logreg.load_state_dict(torch.load("linear1.pt"))
def get_features_from_encoder(encoder, loader):
    
    x_train = []
    y_train = []
    names = []

    # get the features from the pre-trained model
    for batch, y, index in loader:
        index = [int(i) for i in index]
        with torch.no_grad():
            batch = batch.to(device)
            features = encoder(batch)[1]
            x_train.extend(features)
            y_train.extend(y.numpy())
            names.extend(index)

            
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    names = torch.tensor(names)
    return x_train, y_train, names

from itertools import chain
logreg.eval()
encoder.eval()
pre = []
t_file = '/code/PyTorch-BYOL/eval/predicted_glia.txt'
t = open(t_file, "w")
names = []
for batch, y, index in test_loader:
    index = [int(i) for i in index]
    with torch.no_grad():
        batch = batch.to(device)
        features = encoder(batch)[1]
        logits = logreg(features)
        # print(logits)
        predictions = torch.argmax(logits, dim=1)
        # print(predictions)
        pre.extend(predictions)
        names.extend(index)
        glias = np.where(predictions.cpu() == 1)
        for glia in glias[0]:
            t.write(str(index[glia])+',\n')


