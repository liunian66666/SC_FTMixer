import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
import warnings
from tqdm import tqdm  # 🚀 引入 tqdm

warnings.filterwarnings('ignore')

from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate

class Weighted_MSE_Loss(nn.Module):
    def __init__(self, weights=None):
        super(Weighted_MSE_Loss, self).__init__()
        if weights is None:
            self.weights = torch.tensor([1.0, 1.0, 2.0, 5.0, 2.0])
        else:
            self.weights = torch.tensor(weights)

    def forward(self, pred, target):
        sq_error = (pred - target) ** 2
        sample_weight = 1.0 + torch.abs(target) * 0.5 
        device = pred.device
        loss = sq_error * sample_weight * self.weights.to(device)
        return torch.mean(loss)

class Exp_CEMP_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_CEMP_Regression, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        from data_provider.data_factory import data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    # def _select_criterion(self):
    #     return Weighted_MSE_Loss(weights=[1.0, 1.0, 1.5, 4.0, 1.5])
    
    def _select_criterion(self):
        if self.args.use_weighted_loss:
            # 直接把 args 里的列表传给 Weighted_MSE_Loss
            print(f"Using Weighted Loss with weights: {self.args.cemp_weights}")
            return Weighted_MSE_Loss(weights=self.args.cemp_weights)
        else:
            print("Using standard MSE Loss")
            return nn.MSELoss()

    

    # def vali(self, vali_data, vali_loader, criterion):
    #     self.model.eval()
    #     total_loss = []
    #     preds, trues = [], []
        
    #     # 🚀 验证集进度条
    #     vali_bar = tqdm(vali_loader, desc='Validation', leave=False)
        
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y) in enumerate(vali_bar):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #             if len(batch_x.shape) == 2: batch_x = batch_x.unsqueeze(1)
                
    #             outputs = self.model(batch_x, task="cemp")
    #             loss = criterion(outputs, batch_y)
    #             total_loss.append(loss.item())
                
    #             preds.append(outputs.detach().cpu().numpy())
    #             trues.append(batch_y.detach().cpu().numpy())
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        preds, trues = [], []
        
        vali_bar = tqdm(vali_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            # ✨ 修改点 1：接收 4 个变量
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_bar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if len(batch_x.shape) == 2: batch_x = batch_x.unsqueeze(-1)
                
                outputs = self.model(batch_x, task="cemp_regression")
                outputs = outputs.reshape(outputs.shape[0], -1)
                
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
                
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if preds.shape[1] < 4:
            raise ValueError(
                f"CEMP regression output dim mismatch: expected >=4, got {preds.shape[1]}. "
                "Please check model output shape."
            )
        c_fe_true = trues[:, 3]
        c_fe_pred = preds[:, 3]
        r2_c_fe = 1 - np.sum((c_fe_true - c_fe_pred)**2) / (np.sum((c_fe_true - np.mean(c_fe_true))**2) + 1e-7)
        
        avg_loss = np.average(total_loss)
        self.model.train()
        return avg_loss, r2_c_fe  # 返回 loss 和 R2

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.best_vali_loss = float('inf')

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            # 🚀 训练进度条，显示 Epoch 信息
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.args.train_epochs}')
            
            # for i, (batch_x, batch_y) in enumerate(train_bar):
            # 接收全部 4 个返回值，虽然 mark 在回归任务中不被模型使用
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_bar):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if len(batch_x.shape) == 2: batch_x = batch_x.unsqueeze(1)

                outputs = self.model(batch_x, task="cemp_regression")
                outputs = outputs.reshape(outputs.shape[0], -1)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
                
                # 🚀 动态更新进度条右侧的 Loss
                train_bar.set_postfix(loss=f'{loss.item():.5f}')

            train_loss = np.average(train_loss)
            vali_loss, vali_r2 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_r2 = self.vali(test_data, test_loader, criterion)

            if vali_loss < self.best_vali_loss:
                self.best_vali_loss = vali_loss

            print("Epoch: {0} cost: {1:.2f}s | Train Loss: {2:.7f} Vali R2: {3:.4f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, vali_r2))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path): os.makedirs(folder_path)

        self.model.eval()
        # 🚀 测试进度条
        test_bar = tqdm(test_loader, desc='Testing')
        with torch.no_grad():
            # ✨ 修改点 1：接收 4 个变量 (batch_x, batch_y, x_mark, y_mark)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_bar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # ✨ 修改点 2：将 mark 也移至 device (占位符)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # ✨ 修改点 3：维度检查，确保输入是 [Batch, 5000, 1]
                if len(batch_x.shape) == 2: 
                    batch_x = batch_x.unsqueeze(-1) # [B, L] -> [B, L, 1]

                # ✨ 修改点 4：根据模型 forward 签名决定是否传入 marks
                # 如果你的 MVAR_iTransformer_v5 接收 marks，则传入；否则保持原样
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark, task="cemp_regression")
                outputs = outputs.reshape(outputs.shape[0], -1)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # ✨ 修改点 5：逆标准化 (只针对回归的 4 个参数)
        # 注意：如果 preds 是 [N, 4]，scaler 也必须是针对 4 维拟合的
        preds = test_data.scaler_y.inverse_transform(preds)
        trues = test_data.scaler_y.inverse_transform(trues)

         # 🚀 强制将 3 维 [N, 4, 1] 转换为 2 维 [N, 4]
        # 使用 reshape 而不是 squeeze，可以确保即使样本数为 1，维度也是正确的 [1, 4]
        if preds.ndim == 3:
            preds = preds.reshape(preds.shape[0], -1)
        if trues.ndim == 3:
            trues = trues.reshape(trues.shape[0], -1)

        # 打印一下形状进行调试（运行成功后可以删掉）
        print(f"Debug - Preds shape: {preds.shape}, Trues shape: {trues.shape}")

        # 这里根据你实际的回归目标调整
        param_names = ['Teff', 'logg', '[Fe/H]', '[C/Fe]'] 
        individual_metrics = []
        
        print('\n' + '='*60)
        print('🌌 CEMP Regression Physical Results')
        print('='*60)
        
        # ✨ 修改点 6：动态获取参数数量，防止索引越界
        num_params = min(preds.shape[1], len(param_names))
        for i in range(num_params):
            p_pred, p_true = preds[:, i], trues[:, i]
            p_name = param_names[i]
            
            p_mse = np.mean((p_pred - p_true) ** 2)
            p_mae = np.mean(np.abs(p_pred - p_true))
            # 这里的 R² 建议加一个小的 epsilon 防止除零
            r2 = 1 - np.sum((p_true - p_pred)**2) / (np.sum((p_true - np.mean(p_true))**2) + 1e-7)
            
            individual_metrics.append([p_mse, p_mae, r2])
            print('Parameter {:<8} --> MSE: {:>10.4f} | MAE: {:>8.4f} | R²: {:>6.4f}'.format(p_name, p_mse, p_mae, r2))

        overall_mae = np.mean(np.abs(preds - trues))
        overall_mse = np.mean((preds - trues) ** 2)
        print('-'*60)
        print('Overall MAE: {:.4f} | Overall MSE: {:.4f}'.format(overall_mae, overall_mse))

        # 保存逻辑
        log_dir = './results/log_mse_mae'
        # 自动创建文件夹
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, "each"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "All"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "npy_results"), exist_ok=True)
        # 获取当前时间
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        result_filename = os.path.join(log_dir, "each",f"{self.args.model}_{time_str}_results.txt")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 文件名也可以带上时间（可选）
        txt_path = os.path.join(log_dir, f'{self.args.model}_{time_str}_cemp_physical_results.txt')

        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(f'🌌 CEMP Regression Test Results\n')
            f.write(f'⏱️  Test Time: {current_time}\n')  # 写入时间
            f.write(f'📌 Setting: {setting}\n')
            f.write('-' * 50 + '\n')
            
            for i, m in enumerate(individual_metrics):  
                if i < len(param_names):
                    f.write(f'{param_names[i]}: MSE={m[0]:.4f}, MAE={m[1]:.4f}, R2={m[2]:.4f}\n')
            
            f.write('-' * 50 + '\n')
            f.write(f'📊 Overall MAE: {overall_mae:.4f}\n')
            f.write(f'📊 Overall MSE: {overall_mse:.4f}\n')
        
        # 追加到总结果文件
        result_filename = os.path.join(log_dir,"All", f"ALL_CEMP_RESULTS.txt")
        
        with open(result_filename, 'a') as f:
            f.write("===============================================\n")
            f.write(f"Model: {self.args.model} | Time: {current_time}\n")
            f.write(f"Setting: {setting}\n")
            for i, m in enumerate(individual_metrics):  
                if i < len(param_names):
                    f.write(f'{param_names[i]}: MSE={m[0]:.4f}, MAE={m[1]:.4f}, R2={m[2]:.4f}\n')
            f.write(f"MSE: {m[0]:.6f}, MAE: {m[1]:.6f}\n")
            f.write("===============================================\n\n")
        
        folder_path = os.path.join(log_dir, "npy_results",setting)
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path , 'metrics_individual.npy', np.array(individual_metrics))
        np.save(folder_path , 'pred_physical.npy', preds)
        np.save(folder_path , 'true_physical.npy', trues)
        return