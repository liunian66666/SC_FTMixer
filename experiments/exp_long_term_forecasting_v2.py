from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _unpack_batch(self, batch):
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long = batch
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x_long = None
        return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long=None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        if batch_x_long is not None:
            batch_x_long = batch_x_long.float().to(self.device)

        if 'PEMS' in self.args.data or 'Solar' in self.args.data:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], dec_inp],
            dim=1
        ).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_enc_long=batch_x_long)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_enc_long=batch_x_long)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_enc_long=batch_x_long)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_enc_long=batch_x_long)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y, batch_x

    # def vali(self, vali_data, vali_loader, criterion):
    #     total_loss = []
    #     self.model.eval()

    #     vali_pbar = tqdm(vali_loader, desc="Validating", leave=False)

    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_pbar):
    #             outputs, batch_y, _ = self._process_one_batch(
    #                 batch_x, batch_y, batch_x_mark, batch_y_mark
    #             )

    #             pred = outputs.detach().cpu()
    #             true = batch_y.detach().cpu()
    #             loss = criterion(pred, true)

    #             total_loss.append(loss.item())

    #     total_loss = np.average(total_loss) if len(total_loss) > 0 else 0.0
    #     self.model.train()
    #     return total_loss
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        # 1. 这里的 tqdm 建议设置 leave=False，验证完自动清除，不占屏幕
        vali_pbar = tqdm(vali_loader, desc="Validating", leave=False)

        with torch.no_grad():
            # 2. 开启混合精度推理加速，减少显存带宽压力
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                for i, batch in enumerate(vali_pbar):
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long = self._unpack_batch(batch)
                    outputs, batch_y, _ = self._process_one_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long
                    )

                    # 3. 🔥 核心优化：直接在 GPU 上计算 Loss
                    # 不要在这里使用 .cpu()，这会强制让 GPU 等待 CPU，产生极大的 IO 延迟
                    loss = criterion(outputs, batch_y)
                    

                    # 4. 只通过 .item() 将标量数值传回 CPU，这是代价最小的方式
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss) if len(total_loss) > 0 else 0.0
        self.model.train()
        return total_loss

    def _save_resume_checkpoint(
        self,
        path,
        epoch,
        model_optim,
        scaler,
        early_stopping,
        best_vali_loss
        ):
        resume_path = os.path.join(path, 'latest.pth')
        tmp_path = resume_path + '.tmp'

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': model_optim.state_dict(),
            'best_vali_loss': best_vali_loss,
            'early_stopping_counter': getattr(early_stopping, 'counter', 0),
            'early_stopping_best_score': getattr(early_stopping, 'best_score', None),
            'early_stopping_val_loss_min': getattr(early_stopping, 'val_loss_min', np.inf),
            'early_stopping_early_stop': getattr(early_stopping, 'early_stop', False),
        }

        if scaler is not None:
            state['scaler_state_dict'] = scaler.state_dict()

        torch.save(state, tmp_path)
        os.replace(tmp_path, resume_path)

    def _load_resume_checkpoint(
        self,
        path,
        model_optim,
        scaler,
        early_stopping
    ):
        resume_path = os.path.join(path, 'latest.pth')

        if not os.path.exists(resume_path):
            print(f'No resume checkpoint found: {resume_path}')
            return 0, float('inf')

        print(f'Loading resume checkpoint: {resume_path}')
        checkpoint = torch.load(resume_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        model_optim.load_state_dict(checkpoint['optimizer_state_dict'])

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
        early_stopping.best_score = checkpoint.get('early_stopping_best_score', None)
        early_stopping.val_loss_min = checkpoint.get('early_stopping_val_loss_min', np.inf)
        early_stopping.early_stop = checkpoint.get('early_stopping_early_stop', False)

        start_epoch = checkpoint['epoch'] + 1
        best_vali_loss = checkpoint.get('best_vali_loss', float('inf'))

        print(f'Resume from epoch {start_epoch + 1}')
        return start_epoch, best_vali_loss
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # model_optim = self._select_optimizer()
        # criterion = self._select_criterion()

        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()

        # self.best_vali_loss = float('inf')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.best_vali_loss = float('inf')
        start_epoch = 0

        if getattr(self.args, 'resume', False):
            start_epoch, self.best_vali_loss = self._load_resume_checkpoint(
                path,
                model_optim,
                scaler,
                early_stopping
            )


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}", leave=True)

            for i, batch in enumerate(train_pbar):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()

                outputs, batch_y, _ = self._process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long
                )

                loss = criterion(outputs, batch_y)

                # ---- Adaptive-expert auxiliary + KD losses ----
                model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                mixer_pred = getattr(model_ref, 'mixer_prediction', None)
                linear_pred = getattr(model_ref, 'linear_prediction', None)
                expert_alpha = getattr(model_ref, 'expert_alpha_prediction', None)
                student_alpha = getattr(model_ref, 'student_alpha_prediction', None)

                # Read from model first (SC_FTMixer_KD stores defaults),
                # fall back to args, then to safe defaults.
                linear_aux_weight = float(
                    getattr(model_ref, 'linear_aux_weight',
                            getattr(self.args, 'linear_aux_weight', 0.0)))
                mixer_aux_weight = float(
                    getattr(model_ref, 'mixer_aux_weight',
                            getattr(self.args, 'mixer_aux_weight', 0.0)))
                gate_distill_weight = float(
                    getattr(model_ref, 'gate_distill_weight',
                            getattr(self.args, 'gate_distill_weight', 0.0)))

                if mixer_pred is not None and mixer_aux_weight > 0:
                    loss = loss + mixer_aux_weight * criterion(mixer_pred, batch_y)
                if linear_pred is not None and linear_aux_weight > 0:
                    loss = loss + linear_aux_weight * criterion(linear_pred, batch_y)

                if (mixer_pred is not None and linear_pred is not None
                        and expert_alpha is not None and gate_distill_weight > 0):
                    distill_target = student_alpha if student_alpha is not None else expert_alpha
                    segment_len = int(getattr(model_ref, 'expert_gate_segment_len', 24))
                    num_segments = distill_target.size(-1)
                    horizon = num_segments * segment_len

                    delta = (linear_pred - mixer_pred).permute(0, 2, 1)
                    residual = (batch_y - mixer_pred).permute(0, 2, 1)

                    pad = horizon - delta.size(-1)
                    if pad > 0:
                        delta = F.pad(delta, (0, pad), mode='replicate')
                        residual = F.pad(residual, (0, pad), mode='replicate')
                    delta = delta.reshape(delta.size(0), delta.size(1), num_segments, segment_len)
                    residual = residual.reshape(residual.size(0), residual.size(1), num_segments, segment_len)

                    oracle_alpha = ((delta * residual).sum(dim=-1)
                                    / (delta.square().sum(dim=-1) + 1e-6))
                    max_alpha = float(getattr(model_ref.expert_blender, 'max_alpha', 1.0))
                    oracle_alpha = oracle_alpha.clamp(0.0, max_alpha).detach()

                    gate_loss = F.smooth_l1_loss(distill_target, oracle_alpha)
                    loss = loss + gate_distill_weight * gate_loss

                    if student_alpha is not None:
                        consistency_loss = F.mse_loss(expert_alpha, student_alpha.detach())
                        loss = loss + 0.5 * gate_distill_weight * consistency_loss
                
                train_loss.append(loss.item())

                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            

            

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss) if len(train_loss) > 0 else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            

            if vali_loss < self.best_vali_loss:
                self.best_vali_loss = vali_loss

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )

            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            early_stopping(vali_loss, self.model, path)

            self._save_resume_checkpoint(
                path=path,
                epoch=epoch,
                model_optim=model_optim,
                scaler=scaler,
                early_stopping=early_stopping,
                best_vali_loss=self.best_vali_loss
            )

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')

    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(
    #             torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth'),
    #                        map_location=self.device)
    #         )

    #     preds = []
    #     trues = []

    #     folder_path = os.path.join('./test_results', setting)
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     test_pbar = tqdm(test_loader, desc="Testing", leave=True)
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_pbar):
    #             outputs, batch_y, batch_x = self._process_one_batch(
    #                 batch_x, batch_y, batch_x_mark, batch_y_mark
    #             )

    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()

    #             if test_data.scale and self.args.inverse:
    #                 shape = outputs.shape
    #                 outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
    #                 batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

    #             pred = outputs
    #             true = batch_y

    #             preds.append(pred)
    #             trues.append(true)

    #             if i % 20 == 0:
    #                 input_data = batch_x.detach().cpu().numpy()

    #                 if test_data.scale and self.args.inverse:
    #                     shape = input_data.shape
    #                     input_data = test_data.inverse_transform(input_data.squeeze(0)).reshape(shape)

    #                 gt = np.concatenate((input_data[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input_data[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    #     # preds = np.array(preds)
    #     # trues = np.array(trues)
    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)

    #     print('test shape:', preds.shape, trues.shape)

    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    #     print('test shape:', preds.shape, trues.shape)

    #     folder_path = os.path.join('./results', setting)
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))

    #     # with open("result_long_term_forecast.txt", 'a') as f:
    #     #     f.write(setting + "  \n")
    #     #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     #     f.write('\n')
    #     #     f.write('\n')

    #     # ========== 核心修改部分 ==========
    #     # 1. 定义目标目录
    #    # ========== 带时间戳的保存逻辑 ==========
    #     log_dir = "./results/log_mse_mae"
    #     # 2. 如果目录不存在则创建
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)
    #     os.makedirs(os.path.join(log_dir, "each"), exist_ok=True)
    #     os.makedirs(os.path.join(log_dir, "All"), exist_ok=True)
    #     os.makedirs(os.path.join(log_dir, "npy_results"), exist_ok=True)
    #     # 3. 拼接模型名称和固定后缀作为文件名，并组合完整路径
    #     time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    #     result_filename = os.path.join(log_dir, "each", f"{self.args.model}_{time_str}_results.txt")

    #     # 获取当前时间
    #     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    #     # 4. 写入文件（追加模式）
    #     with open(result_filename, 'a') as f:
    #         f.write(setting + "  \n")
    #         f.write(f"Time: {current_time}  \n")  # 新增时间
    #         f.write('mse:{}, mae:{}'.format(mse, mae))
    #         f.write('\n')
    #         f.write('\n')
    #     # ========== 核心修改结束 ==========

    #     result_filename = os.path.join(log_dir, "All",f"{self.args.model}_ALL_forecast_RESULTS.txt")


    #     with open(result_filename, 'a') as f:
    #         f.write("===============================================\n")
    #         f.write(f"Model: {self.args.model} | Time: {current_time}\n")
    #         f.write(f"Setting: {setting}\n")
    #         f.write('mse:{}, mae:{}\n'.format(mse, mae))
    #         f.write("===============================================\n\n")
    #     folder_path = os.path.join(log_dir, "npy_results",setting)
    #     os.makedirs(folder_path, exist_ok=True)
    #     np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(os.path.join(folder_path, 'pred.npy'), preds)
    #     np.save(os.path.join(folder_path,'true.npy'), trues)

    #     return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth'),
                        map_location=self.device)
            )

        preds = []
        trues = []

        # 只有需要画图时才创建目录
        if getattr(self.args, 'visualize', False):
            folder_path = os.path.join('./test_results', setting)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.model.eval()
        test_pbar = tqdm(test_loader, desc="Testing", leave=True)
        
        with torch.no_grad():
            # 1. 开启混合精度推理 (AMP)
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                for i, batch in enumerate(test_pbar):
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long = self._unpack_batch(batch)
                    outputs, batch_y, batch_x = self._process_one_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_long
                    )

                    # 2. 推迟 .cpu() 操作，先在 GPU 上处理必要逻辑
                    # 将数据转为 numpy (注意：如果是为了纯性能测试指标，甚至可以连这一步都推迟到循环外)
                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()

                    # 3. 优化逆标准化逻辑：避免频繁使用 squeeze(0)，改用兼容 batch 的 reshape
                    if test_data.scale and self.args.inverse:
                        shape = pred.shape
                        # 批量处理比单个处理快得多
                        pred = test_data.inverse_transform(pred.reshape(-1, shape[-1])).reshape(shape)
                        true = test_data.inverse_transform(true.reshape(-1, shape[-1])).reshape(shape)

                    preds.append(pred)
                    trues.append(true)

                    # 4. 根据 args.visualize 决定是否画图（这是最慢的操作）
                    if getattr(self.args, 'visualize', False) and i % 20 == 0:
                        input_data = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            input_shape = input_data.shape
                            input_data = test_data.inverse_transform(input_data.reshape(-1, input_shape[-1])).reshape(input_shape)

                        gt = np.concatenate((input_data[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_data[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 5. 循环外一次性合并结果（比在循环内逐个合并快）
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        print('test shape:', preds.shape, trues.shape)

        folder_path = os.path.join('./results/experiment_results', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # with open("result_long_term_forecast.txt", 'a') as f:
        #     f.write(setting + "  \n")
        #     f.write('mse:{}, mae:{}'.format(mse, mae))
        #     f.write('\n')
        #     f.write('\n')

        # ========== 核心修改部分 ==========
        # 1. 定义目标目录
        # ========== 带时间戳的保存逻辑 ==========
        log_dir = "./results/log_mse_mae"
        # 2. 如果目录不存在则创建
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, "each"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "All"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "npy_results"), exist_ok=True)
        # 3. 拼接模型名称和固定后缀作为文件名，并组合完整路径
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        result_filename = os.path.join(log_dir, "each", f"{self.args.model}_{time_str}_results.txt")

        # 获取当前时间
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 4. 写入文件（追加模式）
        with open(result_filename, 'a') as f:
            f.write(f"Args: {self.args}\n")
            f.write(setting + "  \n")
            f.write(f"Time: {current_time}  \n")  # 新增时间
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
        # ========== 核心修改结束 ==========

        result_filename = os.path.join(log_dir, "All",f"{self.args.model}_ALL_forecast_RESULTS.txt")


        with open(result_filename, 'a') as f:
            f.write("===============================================\n")
            f.write(f"Model: {self.args.model} | Time: {current_time}\n")
            f.write(f"Args: {self.args}\n")
            f.write(f"Setting: {setting}\n")
            f.write('mse:{}, mae:{}\n'.format(mse, mae))
            f.write("===============================================\n\n")
        folder_path = os.path.join(log_dir, "npy_results",setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path,'true.npy'), trues)
        return mse, mae
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()

                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)

                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = os.path.join('./results/experiment_results', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)

        return
