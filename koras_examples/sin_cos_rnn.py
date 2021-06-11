import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optimizers


def main(koras):

    class RNN(koras.models.Model):
        def __init__(self, hidden_dim):
            super().__init__()
            self.l1 = nn.RNN(2, hidden_dim,
                             nonlinearity='tanh',
                             batch_first=True)
            self.l2 = nn.Linear(hidden_dim, 2)

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

    def cos(x, T=100):
        return np.cos(2.0 * np.pi * x / T)

    def toy_problem(T=100, ampl=0.05):
        x = np.arange(0, 2*T + 1)
        noise = ampl * np.random.uniform(low=-1.0, high=1.0,
                                         size=len(x))
        return np.array([sin(x) + noise, cos(x) + noise]).reshape(-1, 2)

    T = 100
    f = toy_problem(T).astype(np.float32)
    length_of_sequences = len(f)
    maxlen = 25

    x = []
    t = []

    for i in range(length_of_sequences - maxlen):
        x.append(f[i:i+maxlen])
        t.append(f[i+maxlen])

    x_train, x_test, t_train, t_test = \
        train_test_split(
            np.array(x), np.array(t), test_size=0.2, shuffle=False)
    x_train, x_val, t_train, t_val = \
        train_test_split(x_train, t_train, test_size=0.2, shuffle=False)

    '''
    2. モデルの構築
    '''
    model = RNN(50).to(device)

    '''
    3. モデルの学習
    '''
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['cosine_similarity'])

    es = koras.callbacks.EarlyStopping(patience=10,
                                       verbose=1)

    model.fit(x_train, t_train,
              epochs=1000, batch_size=100,
              verbose=2,
              validation_data=(x_val, t_val),
              callbacks=[es])

    '''
    4. モデルの評価
    '''
    loss, metrics = model.evaluate(x_test, t_test, verbose=0)
    print('test_loss: {:.3f}, test_cosine_similarity: {:.3f}'.format(
        loss,
        metrics['cosine_similarity']
    ))
