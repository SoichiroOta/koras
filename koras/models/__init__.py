from typing import List

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch


LOSS_DICT = {
    'binary_crossentropy': nn.BCELoss
}

METRIC_DICT = {
    'accuracy': lambda t, preds: accuracy_score(t, preds.data.cpu().numpy() > 0.5)
}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def compile(self, optimizer, loss: str, metrics: List[str]):
        self.optimizer = optimizer
        self.loss = LOSS_DICT[loss]()
        self.metrics = metrics

    def _compute_loss(self, t, y):
        return self.loss(y, t)

    def _compute_metrics(self, t, preds):
        return {metric: METRIC_DICT[metric](t, preds) for metric in self.metrics}

    def _train_step(self, x, t):
        self.train()
        preds = self(x)
        loss = self._compute_loss(t, preds)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = self._compute_metrics(t, preds)

        return loss, metrics

    def fit(self, x_train, t_train, epochs: int, batch_size: int, verbose: int = 0, device=None):
        n_batches = x_train.shape[0] // batch_size
        device_ = device if device else self.device

        for epoch in range(epochs):
            train_loss = 0.
            train_metrics = dict()
            x_, t_ = shuffle(x_train, t_train)
            x_ = torch.Tensor(x_).to(device_)
            t_ = torch.Tensor(t_).to(device_)

            for n_batch in range(n_batches):
                start = n_batch * batch_size
                end = start + batch_size
                loss, train_metrics = self._train_step(
                    x_[start:end], t_[start:end])
                train_loss += loss.item()

            if verbose:
                metrics_message = ', ' + ', '.join([
                    '{}: {:.3f}'.format(
                        metric, metric_value
                    ) for metric, metric_value in train_metrics.items()
                ]) if self.metrics else ''
                print('epoch: {}, loss: {:.3}{}'.format(
                    epoch+1,
                    train_loss,
                    metrics_message
                ))

        return self

    def _test_step(self, x, t, device=None):
        device_ = device if device else self.device
        x = torch.Tensor(x).to(device_)
        t = torch.Tensor(t).to(device_)
        self.eval()
        preds = self(x)
        loss = self._compute_loss(t, preds)

        return loss, preds

    def evaluate(self, x_test, t_test, verbose: int = 0):
        loss, preds = self._test_step(x_test, t_test)
        test_loss = loss.item()
        test_metrics = self._compute_metrics(t_test, preds)

        if verbose:
            metrics_message = ', ' + ', '.join([
                'test_{}: {:.3f}'.format(
                    metric, metric_value
                ) for metric, metric_value in test_metrics.items()
            ]) if self.metrics else ''
            print('test_loss: {:.3f}{}'.format(
                test_loss,
                metrics_message
            ))

        return test_loss, test_metrics
