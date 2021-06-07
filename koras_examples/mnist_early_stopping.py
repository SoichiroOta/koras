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

    es = koras.callbacks.EarlyStopping(patience=5, verbose=1)

    hist = model.fit_data_loader(train_dataloader,
                                 epochs=1000,
                                 verbose=2,
                                 val_data_loader=val_dataloader,
                                 callbacks=[es])

    '''
    4. モデルの評価
    '''
    # 検証データの誤差の可視化
    loss = hist['loss']
    val_loss = hist['val_loss']

    fig = plt.figure()
    plt.rc('font', family='serif')
    plt.plot(range(len(loss)), loss,
             color='gray', linewidth=1,
             label='loss')
    plt.plot(range(len(val_loss)), val_loss,
             color='black', linewidth=1,
             label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.savefig('output.jpg')
    plt.show()

    # テストデータの評価
    loss, metrics = model.evaluate_data_loader(test_dataloader, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        metrics['accuracy']
    ))

    exit()

    def test_step(x, t):
        return val_step(x, t)

    test_loss = 0.
    test_acc = 0.

    for (x, t) in test_dataloader:
        x, t = x.to(device), t.to(device)
        loss, preds = test_step(x, t)
        test_loss += loss.item()
        test_acc += \
            accuracy_score(t.tolist(),
                           preds.argmax(dim=-1).tolist())

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        test_loss,
        test_acc
    ))

    exit()

    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. データの準備
    '''
    mnist = datasets.mnist
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)

    x_train, x_val, t_train, t_val = \
        train_test_split(x_train, t_train, test_size=0.2)

    '''
    2. モデルの構築
    '''
    model = Sequential()
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    '''
    3. モデルの学習
    '''
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss',
                       patience=5,
                       verbose=1)

    hist = model.fit(x_train, t_train,
                     epochs=1000, batch_size=100,
                     verbose=2,
                     validation_data=(x_val, t_val),
                     callbacks=[es])

    '''
    4. モデルの評価
    '''
    # 誤差の可視化
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    fig = plt.figure()
    plt.rc('font', family='serif')
    plt.plot(range(len(loss)), loss,
             color='gray', linewidth=1,
             label='loss')
    plt.plot(range(len(val_loss)), val_loss,
             color='black', linewidth=1,
             label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    # plt.savefig('output.jpg')
    plt.show()

    # テストデータの評価
    loss, acc = model.evaluate(x_test, t_test, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        acc
    ))
