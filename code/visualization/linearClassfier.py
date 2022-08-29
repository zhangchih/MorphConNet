import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import json

class LinearNet(nn.Module):
    def __init__(self, n_feature, n_class):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, n_class)
        
    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y
    
def load_data(filepath):
    with open(filepath) as file_obj:
        features = json.load(file_obj)
    feat = []
    cls = []
    for value in features.values():
        feat.append(value['features'][0])
        cls.append(value['class'][0])
    feat = np.array(feat)
    cls = np.array(cls)
    return feat, cls

if __name__ == '__main__':
    train_file = './train_features.json'
    test_file = './features.json'
    batch_size = 10
    num_inputs = 128
    num_epochs = 2000
    num_outputs = 4
    
    data, label = load_data(train_file)
    x_train = torch.from_numpy(data).type(torch.FloatTensor)
    y_train = torch.from_numpy(label).type(torch.LongTensor)
    dataset = Data.TensorDataset(x_train,y_train)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    
    net = LinearNet(num_inputs, num_outputs)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-2)
    for epoch in range(1, num_epochs + 1):
        loss_sum = 0.0
        for step,(x, y) in enumerate(data_iter):
            y_pred = net(x)
            y = y.squeeze()
            loss = loss_func(y_pred, y)
            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: %d, Loss: %f" %(epoch, loss_sum/batch_size))
        
    acc_test_sum = 0.0
    acc_train_sum = 0.0
    data, label = load_data(test_file)
    x_test = torch.from_numpy(data).type(torch.FloatTensor)
    y_test = torch.from_numpy(label).type(torch.LongTensor)
    acc_test_sum += (net(x_test).argmax(dim=1) == y_test.squeeze()).sum()     
    print("test accuracy: %f" % (acc_test_sum / len(x_test))) 
    acc_train_sum += (net(x_train).argmax(dim=1) == y_train.squeeze()).sum()     
    print("train accuracy: %f" % (acc_train_sum / len(x_train))) 