import numpy as np
import torch
import os
import matplotlib.pyplot as plt

# =========================================================================
# 1. 早停机制 (Early Stopping)
# 作用：当验证集 Loss 连续 patience 个 Epoch 没有下降时，自动停止训练，
#       并保存当前表现最好的模型权重，防止严重过拟合。
# =========================================================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        :param patience: 容忍多少个 epoch 验证集 loss 不下降
        :param verbose: 是否打印保存权重的日志信息
        :param delta: 判定为 loss 下降的最小阈值 (通常为0)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # 我们希望 val_loss 越小越好，所以 score 取负数
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """当验证集 loss 达到历史最小时，保存模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # 保存模型权重到指定的 path (通常是 checkpoints/setting_name/checkpoint.pth)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


# =========================================================================
# 2. 动态学习率调整 (Adjust Learning Rate)
# 作用：随着 Epoch 增加，逐渐减小学习率，帮助模型在 Loss 平原上平稳收敛。
# =========================================================================
# def adjust_learning_rate(optimizer, epoch, args):
#     """
#     根据传入的 args.lradj 策略，动态调整优化器的学习率 (Learning Rate)
#     """
#     # 策略 1: type1 (iTransformer 默认策略，每个 epoch 折半)
#     if args.lradj == 'type1':
#         lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        
#     # 策略 2: type2 (固定 step 衰减，常用于图像或复杂回归任务)
#     elif args.lradj == 'type2':
#         lr_adjust = {
#             2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
#             10: 5e-7, 15: 1e-7, 20: 5e-8
#         }
        
#     # 策略 3: half (每 2 个 epoch 折半，较为平缓)
#     elif args.lradj == 'CEMP':
#         lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 2))}
        
#     # 策略 4: cos (余弦退火模拟，暂不启用)
#     else:
#         lr_adjust = {}

#     # 如果当前 epoch 在设定好的衰减字典里，则执行衰减
#     if epoch in lr_adjust.keys():
#         lr = lr_adjust[epoch]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         print('Updating learning rate to {}'.format(lr))

def adjust_learning_rate(optimizer, epoch, args):
    """
    根据传入的 args.lradj 策略，动态调整优化器的学习率 (Learning Rate)
    
    新增策略说明:
    - CEMP_optimized: 针对CEMP任务优化的策略（推荐）
      * Warmup阶段（前5个epoch）：线性增加学习率
      * 主训练阶段：每5个epoch折半（比原CEMP温和）
      * 设置最小学习率下限，防止过小
    """
    # 获取配置参数（带默认值）
    base_lr = args.learning_rate
    min_lr = getattr(args, 'min_lr', 1e-6)  # 最小学习率，默认1e-6
    warmup_epochs = getattr(args, 'warmup_epochs', 5)  # Warmup轮数，默认5
    
    # 策略 1: type1 (iTransformer 默认策略，每个 epoch 折半)
    if args.lradj == 'type1':
        lr_adjust = {epoch: base_lr * (0.5 ** ((epoch - 1) // 1))}
        
    # 策略 2: type2 (固定 step 衰减，常用于图像或复杂回归任务)
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        
    # 策略 3: CEMP (原策略，每 2 个 epoch 折半)
    elif args.lradj == 'CEMP':
        lr_adjust = {epoch: base_lr * (0.5 ** ((epoch - 1) // 2))}
    
    # 🔥 策略 4: CEMP_optimized (推荐用于CEMP任务)
    elif args.lradj == 'CEMP_optimized':
        if epoch <= warmup_epochs:
            # Warmup阶段：线性增加学习率
            lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs)
        else:
            # 主训练阶段：每5个epoch折半（比原CEMP的每2个epoch温和）
            decay_epoch = epoch - warmup_epochs
            lr = base_lr * (0.5 ** (decay_epoch // 5))
            lr = max(lr, min_lr)  # 确保不低于最小学习率
        
        # 直接设置学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'[CEMP_optimized] Epoch {epoch}: lr = {lr:.2e}')
        return  # 直接返回，不走下面的字典逻辑
    
    # 🔥 策略 5: cosine_warmup (余弦退火 + Warmup，理论最优)
    elif args.lradj == 'cosine_warmup':
        total_epochs = getattr(args, 'train_epochs', 100)
        
        if epoch <= warmup_epochs:
            # Warmup阶段
            lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs)
        else:
            # 余弦退火阶段
            cosine_epochs = total_epochs - warmup_epochs
            current_cosine_epoch = epoch - warmup_epochs
            lr = min_lr + 0.5 * (base_lr - min_lr) * (
                1 + np.cos(np.pi * current_cosine_epoch / cosine_epochs)
            )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'[Cosine_Warmup] Epoch {epoch}: lr = {lr:.2e}')
        return
    
    # 🔥 策略 6: step_warmup (阶梯式衰减 + Warmup，稳定可控)
    elif args.lradj == 'step_warmup':
        if epoch <= warmup_epochs:
            # Warmup阶段
            lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs)
        else:
            # 阶梯式衰减：每10个epoch降低到0.5倍
            decay_epoch = epoch - warmup_epochs
            lr = base_lr * (0.5 ** (decay_epoch // 10))
            lr = max(lr, min_lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'[Step_Warmup] Epoch {epoch}: lr = {lr:.2e}')
        return

    # Logistic warmup/slow decay used for very large batches with few
    # optimizer steps per epoch.
    elif args.lradj == 'sigmoid':
        k = 0.5
        smoothing = 10.0
        warmup_center = 10.0
        lr = (
            base_lr / (1.0 + np.exp(-k * (epoch - warmup_center)))
            - base_lr
            / (1.0 + np.exp(-(k / smoothing) * (epoch - warmup_center * smoothing)))
        )
        lr = max(float(lr), 0.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'[Sigmoid] Epoch {epoch}: lr = {lr:.2e}')
        return
        
    # 默认策略（保持原有逻辑）
    else:
        lr_adjust = {}

    # 原有的字典式衰减逻辑（用于type1, type2, CEMP）
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr:.2e}')


# =========================================================================
# 3. 标准化工具 (Standard Scaler) -[补充赠送，写 Dataset 必备]
# 作用：将数据归一化为均值 0，方差 1 (Z-score Normalization)
#       不仅用于输入数据，当你预测 CEMP 时，标签 y 也强烈建议通过它进行归一化。
# =========================================================================
class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        # 加上 1e-5 防止除以 0 导致 Nan
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / (std + 1e-5)

    def inverse_transform(self, data):
        # 测试阶段：将模型的输出还原回真实的物理量级 (比如把 0.1 还原成 5000K 的温度)
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
