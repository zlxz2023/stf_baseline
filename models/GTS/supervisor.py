from tqdm import tqdm

import torch
import numpy as np
from .model import GTSModel
from data import get_dataloader, load_adj
from utils import get_logger, metric, masked_mae, StandardScaler
import pandas as pd
import os
import time
from sklearn.neighbors import kneighbors_graph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GTSSupervisor:
    def __init__(self, **kwargs):
        # init
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.seed = kwargs.get('seed', 2026)
        self.temperature = float(self._model_kwargs.get('temperature'))
        self.opt = self._train_kwargs.get('optimizer')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.ANNEAL_RATE = 0.00003
        self.temp_min = 0.1
        self.epoch_use_regularization = self._train_kwargs.get('epoch_use_regularization')
        dataset_name = os.path.basename(
            os.path.normpath(self._data_kwargs['dataset_dir'])
        )
        self._model_dir = os.path.join('runs/checkpoints/GTS', dataset_name)

        # logging.
        self._log_dir = self._get_log_dir(dataset_name, kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = get_dataloader(self._data_kwargs['dataset_dir'], self._data_kwargs['train_val_test_ratio'], self._model_kwargs['input_len'], self._model_kwargs['output_len'], self._data_kwargs['batch_size'], self._data_kwargs['shuffle'], self._data_kwargs['num_workers'])
        self.standard_scaler = self._data['scaler']

        ### Feas
        # if self._data_kwargs['dataset_dir'] == 'data/METR-LA':
        #     df = pd.read_hdf('./data/metr-la.h5')
        # elif self._data_kwargs['dataset_dir'] == 'data/PEMS-BAY':
        #     df = pd.read_hdf('./data/pems-bay.h5')
        data_path = os.path.join(self._data_kwargs['dataset_dir'], "data.csv")
        df = pd.read_csv(data_path, index_col=0)
        #else:
        #    df = pd.read_csv('./data/pmu_normalized.csv', header=None)
        #    df = df.transpose()
        num_samples = df.shape[0]
        num_train = round(num_samples * self._data_kwargs['train_val_test_ratio'][0])
        df = df.iloc[:num_train].values
        scaler = StandardScaler()
        train_feas = scaler.fit_transform(df)
        self._train_feas = torch.Tensor(train_feas).to(device)
        #print(self._train_feas.shape)

        k = self._train_kwargs.get('knn_k')
        knn_metric = 'cosine'
        g = kneighbors_graph(train_feas.T, k, metric=knn_metric)
        g = np.array(g.todense(), dtype=np.float32)
        self.adj_mx = torch.Tensor(g).to(device)
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.input_len = int(self._model_kwargs.get('input_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.output_len = int(self._model_kwargs.get('output_len', 1))  # for the decoder

        # setup model
        GTS_model = GTSModel(logger=self._logger, **self._model_kwargs)
        self.model = GTS_model.cuda() if torch.cuda.is_available() else GTS_model
        self._logger.info("Model created")

        # 取消继续训练功能
        self._epoch_num = 0

    @staticmethod
    def _get_log_dir(dataset_name, kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            lr = kwargs['train'].get('base_lr')
            output_len = kwargs['model'].get('output_len')
            run_id = 'GTS_%s_%d_lr_%g_bs_%d_%s/' % (
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
        # model_path = os.path.join(self._model_dir, f'epo{epoch}.tar')
        model_path = os.path.join(self._model_dir, f'best_{self.seed}.tar')
        
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

    def evaluate(self, label, batches_seen=0, gumbel_soft=True):
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
            temp = self.temperature

            for batch_idx, (x, y) in enumerate(self._data['val_loader']):
                x, y = self._prepare_data(x, y)

                output, mid_output = self.model(label, x, self._train_feas, temp, gumbel_soft)

                if label == 'without_regularization': 
                    y_true = self.standard_scaler.inverse_transform(y.detach().cpu().numpy())
                    y_pred = self.standard_scaler.inverse_transform(output.detach().cpu().numpy())
                    y_true = torch.as_tensor(y_true, device=device)
                    y_pred = torch.as_tensor(y_pred, device=device)
                    mae, rmse, wmape = metric(y_pred, y_true)
                    wmapes.append(wmape)
                    # mses.append(mse)
                    rmses.append(rmse)
                    losses.append(mae)
                else:
                    loss_1 = self._compute_loss(y, output)
                    pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
                    true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                    compute_loss = torch.nn.BCELoss()
                    loss_g = compute_loss(pred, true_label)
                    loss = loss_1 + loss_g
                    # option
                    # loss = loss_1 + 10*loss_g
                    losses.append((loss_1.item()+loss_g.item()))

                    y_true = self.standard_scaler.inverse_transform(y)
                    y_pred = self.standard_scaler.inverse_transform(output)
                    mae, rmse, wmape = metric(y_pred, y_true)
                    wmapes.append(wmape)
                    rmses.append(rmse)
                    # mses.append(mse)

                #if batch_idx % 100 == 1:
                #    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)
            mean_loss = np.mean(losses)
            mean_wmape = np.mean(wmapes)
            # mean_rmse = np.sqrt(np.mean(mses))
            mean_rmse = np.mean(rmses) #another option

            if label == 'without_regularization':
                return mean_loss, mean_wmape, mean_rmse
            else:
                return mean_loss

    def test(self, gumbel_soft=True):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model = self.model.eval()
    
            all_preds = []
            all_labels = []
            temp = self.temperature
            
            for batch_idx, (x, y) in enumerate(self._data['test_loader']):
                x, y = self._prepare_data(x, y)
                
                output, mid_output = self.model('without_regularization', x, self._train_feas, temp, gumbel_soft)
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                # GTS output and y shape:
                # [output_len, batch, num_nodes * output_dim]
                all_preds.append(y_pred)
                all_labels.append(y_true)
            preds = torch.cat(all_preds, dim=1)
            labels = torch.cat(all_labels, dim=1)
            # 逐步输出损失
            for i in range(preds.shape[1]):
                mae, rmse, wmape = metric(preds[i], labels[i])
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
               steps, patience=200, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        best_model_path = None
        wait = 0
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = len(self._data['train_loader'])
        self._logger.info("num_batches:{}".format(num_batches))

        # 用于继续训练，记录已经见过多少批次的样本
        batches_seen = num_batches * self._epoch_num
        
        # 开始训练
        for epoch_num in range(self._epoch_num, epochs):
            self.model = self.model.train()
            losses = []
            start_time = time.time()
            temp = self.temperature
            gumbel_soft = True

            if epoch_num < self.epoch_use_regularization:
                label = 'with_regularization'
            else:
                label = 'without_regularization'

            for _, (x, y) in enumerate(
                tqdm(
                    self._data['train_loader'],
                    total=num_batches,
                    desc=f"Epoch {epoch_num + 1}/{epochs}",
                )
            ):
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)
                output, mid_output = self.model(label, x, self._train_feas, temp, gumbel_soft, y, batches_seen)
                if (epoch_num % epochs) == epochs - 1:
                    output, mid_output = self.model(label, x, self._train_feas, temp, gumbel_soft, y, batches_seen)

                if batches_seen == 0:
                    if self.opt == 'adam':
                        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)
                    elif self.opt == 'sgd':
                        optimizer = torch.optim.SGD(self.model.parameters(), lr=base_lr)
                    else:
                        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

                self.model.to(device)
                
                #if batch_idx % 100 == 1:
                #    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)

                if label == 'without_regularization':  # or label == 'predictor':
                    loss = self._compute_loss(y, output)
                    losses.append(loss.item())
                else:
                    loss_1 = self._compute_loss(y, output)
                    pred = mid_output.view(mid_output.shape[0] * mid_output.shape[1])
                    true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                    compute_loss = torch.nn.BCELoss()
                    loss_g = compute_loss(pred, true_label)
                    loss = loss_1 + loss_g
                    # option
                    # loss = loss_1 + 10*loss_g
                    losses.append((loss_1.item()+loss_g.item()))

                self._logger.debug(loss.item())
                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                optimizer.step()
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            end_time = time.time()

            if label == 'without_regularization':
                val_loss, val_wmape, val_rmse = self.evaluate(label, batches_seen=batches_seen, gumbel_soft=gumbel_soft)
                end_time2 = time.time()

                if (epoch_num % log_every) == log_every - 1:
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, val_wmape: {:.4f}, val_rmse: {:.4f}, lr: {:.6f}, ' \
                              '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                        np.mean(losses), val_loss, val_wmape, val_rmse,
                                                        lr_scheduler.get_lr()[0],
                                                        (end_time - start_time), (end_time2 - start_time))
                    self._logger.info(message)
            else:
                val_loss = self.evaluate(label, batches_seen=batches_seen, gumbel_soft=gumbel_soft)

                end_time2 = time.time()

                if (epoch_num % log_every) == log_every - 1:
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}'.format(epoch_num, epochs,
                                                                                             batches_seen,
                                                                                             np.mean(losses), val_loss)
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
            self.test(gumbel_soft=True)
        else:
            self._logger.warning('Final test skipped: no best model was saved during training.')


    def _prepare_data(self, x, y):
        """
        :param x: [B, T_in, N, C]
        :param y: [B, T_out, N, C]
        :returns x shape (input_len, batch_size, num_sensor * input_dim)
                 y shape (output_len, batch_size, num_sensor * output_dim)
        """
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, input_len, num_sensor, C)
        :param y: shape (batch_size, output_len, num_sensor, C)
        :returns x shape (input_len, batch_size, num_sensor, input_dim)
                 y shape (output_len, batch_size, num_sensor, output_dim)
        """
        # x = torch.from_numpy(x).float()
        # y = torch.from_numpy(y).float()
        x = x[:, :, :, :self._model_kwargs['input_dim']]
        y = y[:, :, :, :self._model_kwargs['output_dim']]
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (input_len, batch_size, num_sensor, input_dim)
        :param y: shape (output_len, batch_size, num_sensor, output_dim)
        :return: x: shape (input_len, batch_size, num_sensor * input_dim)
                 y: shape (output_len, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.input_len, batch_size, self.num_nodes * self.input_dim)
        y = y.view(self.output_len, batch_size, self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae(y_predicted, y_true, 0.0)
