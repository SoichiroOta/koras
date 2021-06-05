import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optimizers


def main(koras):
    class MLP(koras.models.Model):
        '''
        多層パーセプトロン
        '''

        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden_dim)
            self.a1 = nn.Sigmoid()
            self.l2 = nn.Linear(hidden_dim, output_dim)
            self.a2 = nn.Sigmoid()

            self.layers = [self.l1, self.a1, self.l2, self.a2]

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
    N = 300
    x, t = datasets.make_moons(N, noise=0.3)
    t = t.reshape(N, 1)

    x_train, x_test, t_train, t_test = \
        train_test_split(x, t, test_size=0.2)

    '''
    2. モデルの構築
    '''
    model = MLP(2, 3, 1).to(device)

    '''
    3. モデルの学習
    '''
    optimizer = optimizers.SGD(model.parameters(), lr=0.1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, t_train,
              epochs=100, batch_size=10,
              verbose=1)

    '''
    4. モデルの評価
    '''
    loss, metrics = model.evaluate(x_test, t_test, verbose=1)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        metrics['accuracy']
    ))
