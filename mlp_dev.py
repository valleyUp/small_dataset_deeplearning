import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import optuna
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch import nn
from torch.utils.data import random_split
from torchmetrics.functional import accuracy
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import plot_optimization_history, plot_param_importances

from model import ResNetBinaryClassifier

# 重构网络结构为 LightningModule
class LitModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, dropout_prob, lr, weight_decay):
        super().__init__()

        self.save_hyperparameters()

        self.criterion = nn.CrossEntropyLoss()
        self.model = ResNetBinaryClassifier(
            input_size=input_size,
            layer_sizes=hidden_size,
            dropout_prob=dropout_prob
        )

    def forward(self, x):
        return self.model(x)
    
    def _step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch=batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch=batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y, 'binary')
        self.log('test_acc', acc)
        return acc

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

# 构建 PyTorch Lightning 的数据模块
class LitDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        super().__init__()
        
    def setup(self, stage: str) -> None:
        scaler = StandardScaler()
        dat = pd.read_csv("./diabetes.csv")
        X = torch.tensor(scaler.fit_transform(dat.iloc[:, :-1].to_numpy()), dtype=torch.float32)
        Y = torch.tensor(dat.iloc[:, -1].to_numpy(), dtype=int)
        self.dataset = TensorDataset(X, Y)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [0.6, 0.2, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# 定义目标函数，这个函数将被Optuna调用来优化超参数
def objective(trial):
    # 超参数搜索空间
    dropout_prob = trial.suggest_float('dropout_prob', 0.3, 0.8)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)
    h1 = trial.suggest_int('h1', 32, 512)
    h2 = trial.suggest_int('h2', 32, 512)
    h3 = trial.suggest_int('h3', 32, 512)
    h4 = trial.suggest_int('h4', 32, 512)
    h5 = trial.suggest_int('h5', 32, 512)
    batch_size = trial.suggest_int('batch_size', 8, 64)

    # 创建数据模块
    datamodule = LitDataModule(batch_size=batch_size)

    # 创建模型
    model = LitModel(
        input_size=8,
        hidden_size=[h1, h2, h3, h4, h5],
        dropout_prob=dropout_prob,
        lr=lr,
        weight_decay=weight_decay
    )

    # 创建 PyTorch Lightning Trainer
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir='./'),
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=5,
        max_epochs=500,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),  # 将 Optuna 子试验与 PyTorch Lightning callback 结合
            EarlyStopping(monitor='val_loss', patience=20, mode='min'),
            ModelCheckpoint(monitor='val_loss', mode='min')
        ]
    )

    # 训练
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')

    # 获取验证集上的损失
    test_acc = trainer.callback_metrics['test_acc'].item()

    return test_acc

# 创建一个 Optuna 研究（Study），并找到最佳超参数
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.RandomSampler(),
    pruner=optuna.pruners.HyperbandPruner(),
)
study.optimize(objective, n_trials=200)
history = plot_optimization_history(study)
history.write_image('history.png')
importance = plot_param_importances(study)
importance.write_image('importance.png')

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value:', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
