from typing import List, Dict, Optional
import math

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
import torch.optim as optimizers
import numpy as np
from tqdm import tqdm

from koras.callbacks import EarlyStopping


LOSS_DICT = {
    'binary_crossentropy': nn.BCELoss,
    'categorical_crossentropy': nn.CrossEntropyLoss,
    'mean_squared_error': lambda: nn.MSELoss(reduction='mean')
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
    def __init__(self, input_type=torch.Tensor, output_type=torch.Tensor):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.input_type = input_type
        self.output_type = output_type
        self.optimizer = None
        self.loss = None
        self.metrics = []

    def _get_optimizer(self, optimizer):
        if optimizer and type(optimizer) is not str:
            return optimizer
        elif optimizer == 'sgd':
            return optimizers.SGD(self.parameters(), lr=0.01)
        elif optimizer == 'adam':
            return optimizers.Adam(self.parameters(),
                                   lr=0.001,
                                   betas=(0.9, 0.999), amsgrad=True)
        else:
            return optimizers.SGD(self.parameters(), lr=0.01)

    def _get_loss(self, loss: str):
        return LOSS_DICT[loss]()

    def set_loss(self, loss):
        if type(loss) is str:
            self.loss = self._get_loss(loss)
        else:
            self.loss = loss
        return self

    def set_metrics(self, metrics: List[str]):
        self.metrics = metrics if metrics else []
        return self

    def compile(self, optimizer, loss, metrics: Optional[List[str]] = None):
        self.optimizer = self._get_optimizer(optimizer)
        self.set_loss(loss)
        if type(metrics) is list:
            self.set_metrics(metrics)
        return self

    def _compute_loss(self, t, y):
        return self.loss(y, t)

    def _compute_metrics(self, t, preds):
        preds_ = preds if type(
            preds) is np.ndarray else preds.data.cpu().numpy()
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

    def _init_hist(self) -> Dict[str, List]:
        hist = {'loss': [], 'val_loss': []}
        hist.update({metric: [] for metric in self.metrics})
        hist.update({f'val_{metric}': [] for metric in self.metrics})
        return hist

    def _train(self, n_batch, batch_size, x_, t_, train_loss, train_metrics):
        start = n_batch * batch_size
        end = start + batch_size
        loss, preds = self._train_step(
            x_[start:end],
            t_[start:end]
        )
        train_loss += loss.item()
        train_metrics = {
            metric: train_metrics[metric] + value for metric, value in self._compute_metrics(t_[start:end], preds).items()
        }
        return train_loss, train_metrics

    def fit(self, x_train, t_train, epochs: int, batch_size: int, verbose: int = 1, device=None, validation_data=None, callbacks=None) -> Dict[str, List]:
        hist = self._init_hist()
        if validation_data:
            x_val, t_val = validation_data
        n_batches_train = math.ceil(x_train.shape[0] / batch_size)
        if validation_data:
            n_batches_val = math.ceil(x_val.shape[0] / batch_size)
        device_ = device if device else self.device

        for epoch in range(epochs):
            train_loss = 0.
            val_loss = 0.
            train_metrics = {metric: 0. for metric in self.metrics}
            val_metrics = {metric: 0. for metric in self.metrics}
            x_, t_ = shuffle(x_train, t_train)
            x_ = self.input_type(x_).to(device_)
            t_ = self.output_type(t_).to(device_)

            if verbose == 1:
                for n_batch in tqdm(range(n_batches_train)):
                    train_loss, train_metrics = self._train(
                        n_batch,
                        batch_size,
                        x_,
                        t_,
                        train_loss,
                        train_metrics
                    )
            else:
                for n_batch in range(n_batches_train):
                    train_loss, train_metrics = self._train(
                        n_batch,
                        batch_size,
                        x_,
                        t_,
                        train_loss,
                        train_metrics
                    )

            train_loss /= n_batches_train
            train_metrics = {
                metric: train_metrics[metric] / n_batches_train for metric in self.metrics}

            hist['loss'].append(train_loss)
            for metric, value in train_metrics.items():
                hist[metric].append(value)

            if validation_data:
                for n_batch in range(n_batches_val):
                    start = n_batch * batch_size
                    end = start + batch_size
                    loss, preds = self._val_step(
                        self.input_type(x_val[start:end]).to(device_),
                        self.output_type(t_val[start:end]).to(device_)
                    )
                    val_loss += loss.item()
                    val_metrics = {
                        metric: val_metrics[metric] + value for metric, value in self._compute_metrics(t_val[start:end], preds).items()}

                val_loss /= n_batches_val
                val_metrics = {
                    f'val_{metric}': val_metrics[metric] / n_batches_val for metric in self.metrics}

                hist['val_loss'].append(val_loss)
                for metric, value in val_metrics.items():
                    hist[metric].append(value)

            if not verbose:
                continue

            train_log_message = self._get_log_message(
                train_loss, train_metrics
            )
            if validation_data:
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

            if not callbacks:
                continue

            for callback in callbacks:
                if type(callback) is EarlyStopping and callback(val_loss):  # 早期終了判定
                    return hist

        return hist

    def _get_log_message(self, loss: str, metrics_dict):
        metrics_message = ', ' + ', '.join([
            '{}: {:.3f}'.format(
                metric, metric_value
            ) for metric, metric_value in metrics_dict.items()
        ]) if self.metrics else ''
        return 'loss: {:.3}{}'.format(
            loss,
            metrics_message
        )

    def _train_with_data_loader(self, x, t, device_, train_loss, train_metrics):
        x, t = x.to(device_), t.to(device_)
        loss, preds = self._train_step(x, t)
        train_loss += loss.item()
        train_metrics = {
            metric: train_metrics[metric] + value for metric, value in self._compute_metrics(t, preds).items()
        }
        return train_loss, train_metrics

    def fit_data_loader(self, train_data_loader, epochs: int, verbose: int = 0, device=None, val_data_loader=None, callbacks=None) -> Dict:
        hist = self._init_hist()
        device_ = device if device else self.device

        for epoch in range(epochs):
            train_loss = 0.
            train_metrics = {metric: 0. for metric in self.metrics}
            val_loss = 0.
            val_metrics = {metric: 0. for metric in self.metrics}

            if verbose == 1:
                for (x, t) in tqdm(train_data_loader):
                    train_loss, train_metrics = self._train_with_data_loader(
                        x,
                        t,
                        device_,
                        train_loss,
                        train_metrics
                    )
            else:
                for (x, t) in train_data_loader:
                    train_loss, train_metrics = self._train_with_data_loader(
                        x,
                        t,
                        device_,
                        train_loss,
                        train_metrics
                    )

            train_loss /= len(train_data_loader)
            train_metrics = {
                metric: train_metrics[metric] / len(train_data_loader) for metric in self.metrics}

            hist['loss'].append(train_loss)
            for metric, value in train_metrics.items():
                hist[metric].append(value)

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

            if not callbacks:
                continue

            for callback in callbacks:
                if type(callback) is EarlyStopping and callback(val_loss):  # 早期終了判定
                    return hist

        return hist

    def _test_step(self, x, t):
        return self._val_step(x, t)

    def _print_test_message(self, test_loss, test_metrics):
        metrics_message = ', ' + ', '.join([
            'test_{}: {:.3f}'.format(
                metric, metric_value
            ) for metric, metric_value in test_metrics.items()
        ]) if self.metrics else ''
        print('test_loss: {:.3f}{}'.format(
            test_loss,
            metrics_message
        ))

    def evaluate(self, x_test, t_test, verbose: int = 0, device=None):
        device_ = device if device else self.device
        x = self.input_type(x_test).to(device_)
        t = self.output_type(t_test).to(device_)

        loss, preds = self._test_step(x, t)
        test_loss = loss.item()
        test_metrics = self._compute_metrics(t_test, preds)

        if verbose:
            self._print_test_message(test_loss, test_metrics)

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
            self._print_test_message(test_loss, test_metrics)

        return test_loss, test_metrics

    def save(self, file_path: str):
        torch.save(self.state_dict(),
                   file_path)  # モデルの重みを保存

    def load(self, file_path: str):
        self.load_state_dict(torch.load(file_path))
        return self
