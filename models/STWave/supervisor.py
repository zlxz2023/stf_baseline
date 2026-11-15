from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .model import STWave
from data import get_dataloader, load_Wavelets_Graph
from utils import get_logger, metric, masked_mae, disentangle
import os
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STWaveSupervisor:
    def __init__(self, **kwargs):
        # init
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.seed = kwargs.get('seed', 2026)
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        dataset_name = os.path.basename(
            os.path.normpath(self._data_kwargs['dataset_dir'])
        )
        self._model_dir = os.path.join('runs/checkpoints/STWave', dataset_name)

        # logging.
        self._log_dir = self._get_log_dir(dataset_name, kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = get_dataloader(self._data_kwargs['dataset_dir'], self._data_kwargs['train_val_test_ratio'], self._model_kwargs['input_len'], self._model_kwargs['output_len'], self._data_kwargs['batch_size'], self._data_kwargs['shuffle'], self._data_kwargs['num_workers'])
        self.standard_scaler = self._data['scaler']
        
        ### features
        data_path = os.path.join(self._data_kwargs['dataset_dir'], "data.csv")
        df = pd.read_csv(data_path, index_col=0)
        num_samples = df.shape[0]
        num_train = round(num_samples * self._data_kwargs['train_val_test_ratio'][0])
        data = df.iloc[:num_train].values

        localadj, spawave, temwave = load_Wavelets_Graph(self._data_kwargs['adj_file'], self._data_kwargs['tem_adj_file'], self._model_kwargs['heads']*self._model_kwargs['dims'], data, self._logger, self._model_kwargs['steps_per_day'])

        # setup model
        STWave_model = STWave(
            self._model_kwargs['heads'], self._model_kwargs['dims'], self._model_kwargs['layers'], self._model_kwargs['samples'],
            localadj, spawave, temwave, self._model_kwargs['input_len'], self._model_kwargs['output_len'], self._model_kwargs['steps_per_day']
        )
        self.model = STWave_model.cuda() if torch.cuda.is_available() else STWave_model
        self._logger.info("Model created")

    @staticmethod
    def _get_log_dir(dataset_name, kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            lr = kwargs['train'].get('base_lr')
            output_len = kwargs['model'].get('output_len')
            run_id = 'STWave_%s_%d_lr_%g_bs_%d_%s/' % (
                dataset_name, output_len, lr, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        # if not os.path.exists('models/'):
        #     os.makedirs('models/')
        os.makedirs(self._model_dir, exist_ok=True)
        model_path = os.path.join(self._model_dir, f'best_{self.seed}.pth')
        
        config = dict(self._kwargs)
        config['model_state_dict'] = self.model.state_dict()
        config['epoch'] = epoch
        # torch.save(config, 'models/epo%d.tar' % epoch)
        # self._logger.info("Saved model at {}".format(epoch))
        # return 'models/epo%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model = self.model.eval()

            losses = []
            # mapes = []
            wmapes = []
            rmses = []
            # mses = []

            for batch_idx, (x, y) in enumerate(self._data['val_loader']):
                xl, xh, yl, y, te = self._prepare_data(x, y)
                y_hat, _ = self.model(xl, xh, te)  # [batch_size, output_len, num_nodes, output_dim]
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(y_hat)
                mae, rmse, wmape = metric(y_pred, y_true)
                wmapes.append(wmape)
                rmses.append(rmse)
                losses.append(mae)
            mean_loss = np.mean(losses)
            mean_wmape = np.mean(wmapes)
            mean_rmse = np.mean(rmses)

            return mean_loss, mean_wmape, mean_rmse

    def test(self):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model = self.model.eval()
    
            all_preds = []
            all_labels = []
            
            for batch_idx, (x, y) in enumerate(self._data['test_loader']):
                xl, xh, yl, y, te = self._prepare_data(x, y)
                y_hat, _ = self.model(xl, xh, te)  # [batch_size, output_len, num_nodes, output_dim]
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(y_hat)
                all_preds.append(y_pred)
                all_labels.append(y_true)
            preds = torch.cat(all_preds, dim=0)
            labels = torch.cat(all_labels, dim=0)
            # 逐步输出损失
            for i in range(preds.shape[1]):
                mae, rmse, wmape = metric(preds[:, i, :, :], labels[:, i, :, :])
                self._logger.info(
                    "Horizon %d, MAE: %.4f, WMAPE: %.4f, RMSE: %.4f",
                    i + 1,
                    mae,
                    wmape,
                    rmse,
                )
            # 输出整个窗口内的损失
            mae, rmse, wmape = metric(preds, labels)
            self._logger.info(
                "All horizons, MAE: %.4f, WMAPE: %.4f, RMSE: %.4f",
                mae,
                wmape,
                rmse,
            )
    

    def _train(self, base_lr,
               patience=200, epochs=100, log_every=1, save_model=1, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        best_model_path = None
        wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                        mode='min', factor=0.1, patience=20,    
                                        threshold=0.001, threshold_mode='rel',
                                        cooldown=0, min_lr=2e-6, eps=1e-08)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = len(self._data['train_loader'])
        self._logger.info("num_batches:{}".format(num_batches))
        
        # 开始训练
        for epoch_num in range(epochs):
            self.model = self.model.train()
            losses = []
            start_time = time.time()

            for _, (x, y) in enumerate(
                tqdm(
                    self._data['train_loader'],
                    total=num_batches,
                    desc=f"Epoch {epoch_num + 1}/{epochs}",
                )
            ):
                optimizer.zero_grad()
                xl, xh, yl, y, te = self._prepare_data(x, y)
                y_hat, y_hat_l = self.model(xl, xh, te)  # [batch_size, output_len, num_nodes, output_dim]
                loss = self._compute_loss(y, y_hat) + self._compute_loss(yl, y_hat_l)
                losses.append(loss.item())
                self._logger.debug(loss.item())
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

            self._logger.info("epoch complete")
            self._logger.info("evaluating now!")
            end_time = time.time()

            val_loss, val_wmape, val_rmse = self.evaluate()
            lr_scheduler.step(val_loss)
            end_time2 = time.time()

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_mae: {:.4f}, val_wmape: {:.4f}, val_rmse: {:.4f}, ' \
                          '{:.1f}s, {:.1f}s'.format(epoch_num, epochs,
                                                    np.mean(losses), val_loss, val_wmape, val_rmse,
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    best_model_path = model_file_name
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

        if best_model_path is not None:
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self._logger.info(
                'Final test: loaded best model from {} (epoch {}, val_loss {:.4f})'.format(
                    best_model_path, checkpoint['epoch'], min_val_loss))
            self.test()
        else:
            self._logger.warning('Final test skipped: no best model was saved during training.')


    def _prepare_data(self, x, y):
        """
        input type: tensor
        :param x: [B, T_in, N, C]
        :param y: [B, T_out, N, C]
        :returns xl shape [B, T_in, N, 1]
                 xh shape [B, T_in, N, 1]
                 yl shape [B, T_out, N, 1]
                 y shape [B, T_out, N, 1]
                 te shape [B, T_in+T_out, 2]
        """
        # 拼接时间特征
        te = torch.cat([x[..., -2:], y[..., -2:]], dim=1)   # [B, T_in+T_out, N, 2]
        te = te.flip(-1)  # [dow, tod]
        te = te[:, :, 0, :]
        # 将x, y逆标准化
        x[..., :1] = self.standard_scaler.inverse_transform(x[..., :1])
        y[..., :1] = self.standard_scaler.inverse_transform(y[..., :1])
        # 转为ndarray
        x = x.numpy()
        y = y.numpy()
        # 执行小波分解
        xl, xh = disentangle(x[..., :1], self._train_kwargs['wave'], self._train_kwargs['level'])
        yl, _ = disentangle(y[..., :1], self._train_kwargs['wave'], self._train_kwargs['level'])
        # 重新标准化
        xl = self.standard_scaler.transform(xl)
        xh = self.standard_scaler.transform(xh)
        yl = self.standard_scaler.transform(yl)
        y[..., :1] = self.standard_scaler.transform(y[..., :1])
        y = y[..., :1]
        xl, xh, yl, y = torch.as_tensor(xl, dtype=torch.float32), torch.as_tensor(xh, dtype=torch.float32), torch.as_tensor(yl, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        return xl.to(device), xh.to(device), yl.to(device), y.to(device), te.to(device)


    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae(y_predicted, y_true, 0.0)