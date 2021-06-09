import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optimizers


def main(koras):

    class RNN(koras.models.Model):
        def __init__(self, hidden_dim):
            super().__init__()
            self.l1 = nn.RNN(1, hidden_dim,
                             nonlinearity='tanh',
                             batch_first=True)
            self.l2 = nn.Linear(hidden_dim, 1)

            nn.init.xavier_normal_(self.l1.weight_ih_l0)
            nn.init.orthogonal_(self.l1.weight_hh_l0)

        def forward(self, x):
            h, _ = self.l1(x)
            y = self.l2(h[:, -1])
            return y

    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1. データの準備
    '''
    def sin(x, T=100):
        return np.sin(2.0 * np.pi * x / T)

    def toy_problem(T=100, ampl=0.05):
        x = np.arange(0, 2*T + 1)
        noise = ampl * np.random.uniform(low=-1.0, high=1.0,
                                         size=len(x))
        return sin(x) + noise

    T = 100
    f = toy_problem(T).astype(np.float32)
    length_of_sequences = len(f)
    maxlen = 25

    x = []
    t = []

    for i in range(length_of_sequences - maxlen):
        x.append(f[i:i+maxlen])
        t.append(f[i+maxlen])

    x = np.array(x).reshape(-1, maxlen, 1)
    t = np.array(t).reshape(-1, 1)

    x_train, x_val, t_train, t_val = \
        train_test_split(x, t, test_size=0.2, shuffle=False)

    '''
    2. モデルの構築
    '''
    model = RNN(50).to(device)

    '''
    3. モデルの学習
    '''
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    es = koras.callbacks.EarlyStopping(patience=10,
                                       verbose=1)

    hist = model.fit(x_train, t_train,
                     epochs=1000, batch_size=100,
                     verbose=2,
                     validation_data=(x_val, t_val),
                     callbacks=[es])

    '''
    4. モデルの評価
    '''
    model.eval()

    # sin波の予測
    sin = toy_problem(T, ampl=0.)
    gen = [None for i in range(maxlen)]

    z = x[:1]

    for i in range(length_of_sequences - maxlen):
        z_ = torch.Tensor(z[-1:]).to(device)
        preds = model(z_).data.cpu().numpy()
        z = np.append(z, preds)[1:]
        z = z.reshape(-1, maxlen, 1)
        gen.append(preds[0, 0])

    # 予測値を可視化
    fig = plt.figure()
    plt.rc('font', family='serif')
    plt.xlim([0, 2*T])
    plt.ylim([-1.5, 1.5])
    plt.plot(range(len(f)), sin,
             color='gray',
             linestyle='--', linewidth=0.5)
    plt.plot(range(len(f)), gen,
             color='black', linewidth=1,
             marker='o', markersize=1, markerfacecolor='black',
             markeredgecolor='black')
    # plt.savefig('output.jpg')
    plt.show()
