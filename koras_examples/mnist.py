import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms


def main(koras):

    class DNN(koras.models.Model):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.a1 = nn.Sigmoid()
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.a2 = nn.Sigmoid()
            self.l3 = nn.Linear(hidden_dim, hidden_dim)
            self.a3 = nn.Sigmoid()
            self.l4 = nn.Linear(hidden_dim, output_dim)

            self.layers = [self.l1, self.a1,
                           self.l2, self.a2,
                           self.l3, self.a3,
                           self.l4]

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)

            return x

    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1. データの準備
    '''
    root = os.path.join('~', '.torch', 'mnist')
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: x.view(-1)])
    mnist_train = datasets.MNIST(root=root,
                                 download=True,
                                 train=True,
                                 transform=transform)
    mnist_test = datasets.MNIST(root=root,
                                download=True,
                                train=False,
                                transform=transform)

    train_dataloader = DataLoader(mnist_train,
                                  batch_size=100,
                                  shuffle=True)
    test_dataloader = DataLoader(mnist_test,
                                 batch_size=100,
                                 shuffle=False)

    '''
    2. モデルの構築
    '''
    model = DNN(784, 200, 10).to(device)

    '''
    3. モデルの学習
    '''
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_data_loader(train_dataloader,
                          epochs=30,
                          verbose=2)

    '''
    4. モデルの評価
    '''
    loss, metrics = model.evaluate_data_loader(test_dataloader, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        metrics['accuracy']
    ))
