import os

import torch
import yaml
from data.multi_view_data_injector import MultiViewDataInjector
from data_utils.ModelNetDataLoader_ori import ModelNetDataLoader
from models.mlp_head import MLPHead
from models.pointnetv2_encoder import PointNetV2
from trainer import BYOLTrainer

print(torch.__version__)
torch.manual_seed(0)


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")


    train_dataset = ModelNetDataLoader('/data/nerdxie/neuron_cluster/neuron_dataset_add_all',split='train', uniform=True, normal_channel=True,)
    print(len(train_dataset))
    # online network
    online_network = PointNetV2(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = PointNetV2(**config['network']).to(device)

#     optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
#                                 **config['optimizer']['params'])
    optimizer = torch.optim.Adam(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=torch.nn.DataParallel(online_network),
                          target_network=torch.nn.DataParallel(target_network),
                          optimizer=optimizer,
                          predictor=torch.nn.DataParallel(predictor),
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
