import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter
from models.kmean import KMEANS

from utils import _create_model_training_folder

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, log_dir='/output/logs', **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter(log_dir)
        self.m = params['m']
        self.log_dir = log_dir
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        # build the queue
        self.queue = None
        self.queue_length = params['queue_length'] - params['queue_length'] % (params['batch_size'])
        self.epoch_queue_starts = params['epoch_queue_starts']
        self.feature_dim = params['feature_dim']
        self.alpha = params['alpha']
        self.kmeans = KMEANS()
        self.freq_update_cluster = params['freq_update_cluster']
        _create_model_training_folder(log_dir, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])


    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
    
    @staticmethod
    def cluster_loss(features, centers, labels):
        logits = 0
        if labels is None:
            return 0
        for i, sample in enumerate(features):
            sim = torch.cosine_similarity(sample.unsqueeze(0), centers, eps=1e-8)
            if (sim != sim).sum() != 0:
                break
            logits += torch.log(torch.softmax(sim / 0.05, dim=0)[labels[i]])
        return -logits

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.log_dir, 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):
            
            # optionally starts a queue
            if self.queue_length > 0 and epoch_counter >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    self.queue_length,
                    self.feature_dim,
                ).cuda()

            for (batch_view_1, batch_view_2), _ in train_loader:

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                
                if niter % self.freq_update_cluster == 0:
                    self.kmeans.fit(self.queue)
                    
                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1

            print("End of epoch {}".format(epoch_counter))
            # save checkpoints
            self.save_model(os.path.join(model_checkpoints_folder, 'model_epoch_{}.pth'.format(epoch_counter)))

        

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1)[0])
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2)[0])
        
        
        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)[0]
            targets_to_view_1 = self.target_network(batch_view_2)[0]
            
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        
        # time to use the queue
        bs = self.batch_size
        self.queue[2*bs:] = self.queue[:-2*bs, :].clone()
        self.queue[:bs] = targets_to_view_1
        self.queue[bs:2*bs] = targets_to_view_2
        
        labels = self.kmeans.find_closest_centers(predictions_from_view_1)
        cluster_loss = self.alpha * self.cluster_loss(predictions_from_view_1, self.kmeans.stable_centers, labels)
        labels = self.kmeans.find_closest_centers(predictions_from_view_2)
        cluster_loss += self.alpha * self.cluster_loss(predictions_from_view_2, self.kmeans.stable_centers, labels)
        return loss.mean() + cluster_loss

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
