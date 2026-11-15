from tqdm import tqdm

import torch
import numpy as np
from .model import STAEformer
from data import get_dataloader, load_adj
from utils import get_logger, metric, masked_mae
import os
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STAEformerSupervisor:
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
        self._model_dir = os.path.join('runs/checkpoints/STAEformer', dataset_name)

        # logging.
        self._log_dir = self._get_log_dir(dataset_name, kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = get_dataloader(self._data_kwargs['dataset_dir'], self._data_kwargs['train_val_test_ratio'], self._model_kwargs['input_len'], self._model_kwargs['output_len'], self._data_kwargs['batch_size'], self._data_kwargs['shuffle'], self._data_kwargs['num_workers'])
        self.standard_scaler = self._data['scaler']
        # _, _, adj = load_adj(self._data_kwargs['graph_pkl_filename'], self._data_kwargs['adjtype'])

        # setup model
        STAEformer_model = STAEformer(
            num_nodes=self._model_kwargs['num_nodes'],
            in_steps=self._model_kwargs['input_len'],
            out_steps=self._model_kwargs['output_len'],
            steps_per_day=self._model_kwargs['steps_per_day'],
            input_dim=self._model_kwargs['input_dim'],
            output_dim=self._model_kwargs['output_dim'],
            input_embedding_dim=self._model_kwargs['input_embedding_dim'],
            tod_embedding_dim=self._model_kwargs['tod_embedding_dim'],
            dow_embedding_dim=self._model_kwargs['dow_embedding_dim'],
            spatial_embedding_dim=self._model_kwargs['spatial_embedding_dim'],
            adaptive_embedding_dim=self._model_kwargs['adaptive_embedding_dim'],
            feed_forward_dim=self._model_kwargs['feed_forward_dim'],
            num_heads=self._model_kwargs['num_heads'],
            num_layers=self._model_kwargs['num_layers'],
            dropout=self._train_kwargs['dropout'],
        )
        self.model = STAEformer_model.cuda() if torch.cuda.is_available() else STAEformer_model
        self._logger.info("Model created")

    @staticmethod
    def _get_log_dir(dataset_name, kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            lr = kwargs['train'].get('base_lr')
            output_len = kwargs['model'].get('output_len')
            run_id = 'STAEformer_%s_%d_lr_%g_bs_%d_%s/' % (
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
                x, y = self._prepare_data(x, y)
                output = self.model(x)  # [batch_size, output_len, num_nodes, output_dim]
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
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
                x, y = self._prepare_data(x, y)
                output = self.model(x)  # [batch_size, output_len, num_nodes, output_dim]
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
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
               steps, patience=200, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1, 
               weight_decay=0.0001, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        best_model_path = None
        wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay, eps=epsilon)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=steps,
            gamma=lr_decay_ratio,
        )

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
                x, y = self._prepare_data(x, y)
                output = self.model(x)  # [batch_size, output_len, num_nodes, output_dim]
                loss = self._compute_loss(y, output)
                losses.append(loss.item())
                self._logger.debug(loss.item())
                loss.backward()

                # gradient clipping - this does it in place
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            end_time = time.time()

            val_loss, val_wmape, val_rmse = self.evaluate()
            end_time2 = time.time()

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] train_mae: {:.4f}, val_mae: {:.4f}, val_wmape: {:.4f}, val_rmse: {:.4f}, ' \
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
        :param x: [B, T_in, N, C]
        :param y: [B, T_out, N, C]
        :returns x shape [B, T_in, N, input_dim]
                 y shape [B, T_out, N, output_dim]
        """
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, input_len, num_sensor, C)
        :param y: shape (batch_size, output_len, num_sensor, C)
        :returns x shape (batch_size, input_len, num_sensor, input_dim)
                 y shape (batch_size, output_len, num_sensor, output_dim)
        """
        # x = torch.from_numpy(x).float()
        # y = torch.from_numpy(y).float()
        x = x[:, :, :, :self._model_kwargs['input_dim']]
        y = y[:, :, :, :self._model_kwargs['output_dim']]
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (batch_size, input_dim, num_sensor, input_len)
        :param y: shape (batch_size, output_dim, num_sensor, output_len)
        :return: x: shape (batch_size, input_dim, num_sensor, input_len)
                 y: shape (batch_size, output_dim, num_sensor, output_len)
        """
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae(y_predicted, y_true, 0.0)
