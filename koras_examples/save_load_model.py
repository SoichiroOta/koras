import os
import numpy as np
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
            self.b1 = nn.BatchNorm1d(hidden_dim)
            self.a1 = nn.ReLU()
            self.d1 = nn.Dropout(0.5)
            self.l2 = nn.Linear(hidden_dim, output_dim)

            self.layers = [self.l1, self.b1, self.a1, self.d1,
                           self.l2]

            for layer in self.layers:
                if type(layer) == nn.Linear:
                    nn.init.kaiming_normal_(layer.weight)

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
    3. モデルの学習・保存
    '''
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_data_loader(train_dataloader,
                          epochs=10,
                          verbose=2,
                          val_data_loader=val_dataloader)

    model.save('models/model_torch.h5')  # モデルを保存

    print('model weights saved to: {}'.format('models/model_torch.h5'))

    '''
    4. モデルの読み込み・評価
    '''
    del model  # これまで学習していたモデルを削除

    model = DNN(784, 200, 10).to(device)  # 新しいモデルを初期化
    model.load('models/model_torch.h5')  # 学習済モデルの重みを設定

    print('-' * 20)
    print('model loaded.')

    # テストデータの評価
    model.set_loss('categorical_crossentropy').set_metrics(['accuracy'])
    loss, metrics = model.evaluate_data_loader(test_dataloader, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        metrics['accuracy']
    ))
