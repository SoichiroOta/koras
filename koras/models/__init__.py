from typing import List, Dict

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import torch.optim as optimizers
import numpy as np


LOSS_DICT = {
    'binary_crossentropy': nn.BCELoss,
    'categorical_crossentropy': nn.CrossEntropyLoss
}

METRIC_DICT = {
    'accuracy': lambda t, preds: accuracy_score(
        t.tolist(),
        preds.argmax(axis=-1).tolist()
    ) if preds.ndim > 1 and preds.shape[1] > 1 else accuracy_score(
        t,
        preds > 0.5
    )
}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def _get_optimizer(self, optimizer):
        if optimizer and type(optimizer) is not str:
            return optimizer
        elif optimizer == 'sgd':
            return optimizers.SGD(self.parameters(), lr=0.01)
        else:
            return optimizers.SGD(self.parameters(), lr=0.01)

    def compile(self, optimizer, loss: str, metrics: List[str]):
        self.optimizer = self._get_optimizer(optimizer)
        self.loss = LOSS_DICT[loss]()
        self.metrics = metrics

    def _compute_loss(self, t, y):
        return self.loss(y, t)

    def _compute_metrics(self, t, preds):
        preds_ = preds.data.cpu().numpy() if type(preds) is not np.ndarray else preds
        return {metric: METRIC_DICT[metric](t, preds_) for metric in self.metrics}

    def _train_step(self, x, t):
        self.train()
        preds = self(x)
        loss = self._compute_loss(t, preds)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, preds

    def _val_step(self, x, t):
        self.eval()
        preds = self(x)
        loss = self._compute_loss(t, preds)

        return loss, preds

    def fit(self, x_train, t_train, epochs: int, batch_size: int, verbose: int = 0, device=None):
        n_batches = x_train.shape[0] // batch_size
        device_ = device if device else self.device

        for epoch in range(epochs):
            train_loss = 0.
            train_metrics = {metric: 0. for metric in self.metrics}
            x_, t_ = shuffle(x_train, t_train)
            x_ = torch.Tensor(x_).to(device_)
            t_ = torch.Tensor(t_).to(device_)

            for n_batch in range(n_batches):
                start = n_batch * batch_size
                end = start + batch_size
                loss, preds = self._train_step(
                    x_[start:end], t_[start:end])
                train_loss += loss.item()
                train_metrics = {
                    metric: train_metrics[metric] + value for metric, value in self._compute_metrics(t_[start:end], preds).items()}

            train_loss /= n_batches
            train_metrics = {
                metric: train_metrics[metric] / n_batches for metric in self.metrics}

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

    def _get_log_message(self, loss, metrics_dict):
        metrics_message = ', ' + ', '.join([
            '{}: {:.3f}'.format(
                metric, metric_value
            ) for metric, metric_value in metrics_dict.items()
        ]) if self.metrics else ''
        return 'loss: {:.3}{}'.format(
            loss,
            metrics_message
        )

    def fit_data_loader(self, train_data_loader, epochs: int, verbose: int = 0, device=None, val_data_loader=None):
        hist = {f'val_{metric}': [] for metric in self.metrics}
        hist['val_loss'] = []
        device_ = device if device else self.device

        for epoch in range(epochs):
            train_loss = 0.
            train_metrics = {metric: 0. for metric in self.metrics}
            val_loss = 0.
            val_metrics = {metric: 0. for metric in self.metrics}

            for (x, t) in train_data_loader:
                x, t = x.to(device_), t.to(device_)
                loss, preds = self._train_step(x, t)
                train_loss += loss.item()
                train_metrics = {
                    metric: train_metrics[metric] + value for metric, value in self._compute_metrics(t, preds).items()}

            train_loss /= len(train_data_loader)
            train_metrics = {
                metric: train_metrics[metric] / len(train_data_loader) for metric in self.metrics}

            if val_data_loader:
                for (x, t) in val_data_loader:
                    x, t = x.to(device), t.to(device)
                    loss, preds = self._val_step(x, t)
                    val_loss += loss.item()
                    val_metrics = {
                        metric: val_metrics[metric] + value for metric, value in self._compute_metrics(t, preds).items()}

                val_loss /= len(val_data_loader)
                val_metrics = {
                    f'val_{metric}': val_metrics[metric] / len(val_data_loader) for metric in self.metrics}

                hist['val_loss'].append(val_loss)
                for metric, value in val_metrics.items():
                    hist[metric].append(value)

            if not verbose:
                continue

            train_log_message = self._get_log_message(
                train_loss, train_metrics
            )
            if val_data_loader:
                val_log_message = self._get_log_message(
                    val_loss, val_metrics
                )
                print('epoch: {}, {}, val_{}'.format(
                    epoch+1,
                    train_log_message,
                    val_log_message
                ))
            else:
                print('epoch: {}, {}'.format(
                    epoch+1,
                    train_log_message
                ))

        return hist

    def _test_step(self, x, t):
        return self._val_step(x, t)

    def evaluate(self, x_test, t_test, verbose: int = 0, device=None):
        device_ = device if device else self.device
        x = torch.Tensor(x_test).to(device_)
        t = torch.Tensor(t_test).to(device_)

        loss, preds = self._test_step(x, t)
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

    def evaluate_data_loader(self, test_data_loader, verbose: int = 0, device=None):
        test_loss = 0.
        test_metrics = {metric: 0. for metric in self.metrics}
        device_ = device if device else self.device

        for (x, t) in test_data_loader:
            x, t = x.to(device_), t.to(device_)
            loss, preds = self._test_step(x, t)
            test_loss += loss.item()
            test_metrics = {
                metric: test_metrics[metric] + value for metric, value in self._compute_metrics(t, preds).items()}

        test_loss /= len(test_data_loader)
        test_metrics = {
            metric: test_metrics[metric] / len(test_data_loader) for metric in self.metrics}

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
