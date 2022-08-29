import os

import torch
import yaml
from data.multi_view_data_injector import MultiViewDataInjector
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.TestDataLoader import TestDataLoader
from models.mlp_head import MLPHead
from models.pointnetv2_encoder import PointNetV2
from trainer import BYOLTrainer

from torch.utils.data.dataloader import DataLoader

import json

print(torch.__version__)
torch.manual_seed(0)


def main():
    save_json = 'train_features.json'
    
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")


    test_dataset = TestDataLoader('/data/nerdxie/neuron_cluster/neuron_dataset_1',split='test', uniform=True, normal_channel=True,istrain=False,)

    # online network
    online_network = PointNetV2(**config['network']).to(device)
    online_network.eval()
#     pretrained_folder = "/output/logs"
    pretrained_folder = "./"
    
    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join(pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model_epoch_0.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
            
    test_loader = DataLoader(test_dataset, batch_size=config['testing']['batch_size'],
                                  num_workers=config['testing']['num_workers'], drop_last=False, shuffle=False)
    online_network.eval()
    
    res = {}
    for batch, target_class, index in test_loader:
        batch = batch.to(device)
        features = online_network(batch)[0]
        features = features.cpu().detach().numpy()
        res[index[0]] = {'class': target_class.detach().numpy().tolist(), 
                   'features': features.tolist()}
    jsres = json.dumps(res)
    file = open(save_json, 'w')
    file.write(jsres)
    file.close()
    

if __name__ == '__main__':
    main()
