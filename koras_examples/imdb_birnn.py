import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optimizers
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main(koras):

    class BiRNN(koras.models.Model):

        def __init__(self, num_words, hidden_dim):
            super().__init__(input_type=torch.LongTensor)
            self.emb = nn.Embedding(num_words, hidden_dim, padding_idx=0)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim,
                                batch_first=True,
                                bidirectional=True)
            self.linear = nn.Linear(hidden_dim*2, 1)
            self.sigmoid = nn.Sigmoid()

            nn.init.xavier_normal_(self.lstm.weight_ih_l0)
            nn.init.orthogonal_(self.lstm.weight_hh_l0)
            nn.init.xavier_normal_(self.linear.weight)

        def forward(self, x):
            h = self.emb(x)
            h, _ = self.lstm(h)
            h = self.linear(h[:, -1])
            y = self.sigmoid(h)
            return y.squeeze()  # (batch_size, 1) => (batch_size,)

    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1. データの準備
    '''
    num_words = 20000
    maxlen = 80

    imdb = datasets.imdb
    word_index = imdb.get_word_index()

    (x_train, t_train), (x_test, t_test) = imdb.load_data(num_words=num_words,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)

    x_train, x_val, t_train, t_val = \
        train_test_split(x_train, t_train, test_size=0.2)

    x_train = pad_sequences(x_train, maxlen=maxlen, padding='pre')
    x_val = pad_sequences(x_val, maxlen=maxlen, padding='pre')
    x_test = pad_sequences(x_test, maxlen=maxlen, padding='pre')

    '''
    2. モデルの構築
    '''
    model = BiRNN(num_words, 128).to(device)

    '''
    3. モデルの学習
    '''
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    es = koras.callbacks.EarlyStopping(patience=5,
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
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        metrics['accuracy']
    ))
