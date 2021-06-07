import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
import torchvision.transforms as transforms


def main(koras):

    class DNN(koras.models.Model):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.a1 = nn.ReLU()
            self.d1 = nn.Dropout(0.5)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
            self.a2 = nn.ReLU()
            self.d2 = nn.Dropout(0.5)
            self.l3 = nn.Linear(hidden_dim, hidden_dim)
            self.a3 = nn.ReLU()
            self.d3 = nn.Dropout(0.5)
            self.l4 = nn.Linear(hidden_dim, output_dim)

            self.layers = [self.l1, self.a1, self.d1,
                           self.l2, self.a2, self.d2,
                           self.l3, self.a3, self.d3,
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

    n_samples = len(mnist_train)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train

    mnist_train, mnist_val = \
        random_split(mnist_train, [n_train, n_val])

    train_dataloader = DataLoader(mnist_train,
                                  batch_size=100,
                                  shuffle=True)
    val_dataloader = DataLoader(mnist_val,
                                batch_size=100,
                                shuffle=False)
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
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit_data_loader(train_dataloader,
                                 epochs=100,
                                 verbose=2,
                                 val_data_loader=val_dataloader)

    '''
    4. モデルの評価
    '''
    # 検証データの誤差の可視化
    val_loss = hist['val_loss']

    fig = plt.figure()
    plt.rc('font', family='serif')
    plt.plot(range(len(val_loss)), val_loss,
             color='black', linewidth=1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.savefig('output.jpg')
    plt.show()

    # 正解率を可視化する場合
    # val_acc = hist['val_accuracy']
    #
    # fig = plt.figure()
    # plt.rc('font', family='serif')
    # plt.plot(range(len(val_acc)), val_acc,
    #          color='black', linewidth=1)
    # plt.xlabel('epochs')
    # plt.ylabel('acc')
    # plt.savefig('output_acc.jpg')
    # plt.show()

    # テストデータの評価
    loss, metrics = model.evaluate_data_loader(test_dataloader, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        metrics['accuracy']
    ))
