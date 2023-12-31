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

# 重构网络结构为 LightningModule
class LitModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob, lr, weight_decay):
        super().__init__()

        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y, 'binary')
        self.log('val_loss', loss)
        self.log('val_acc', acc, on_epoch=True)
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
        
    #def prepare_data(self) -> None:
        
    def setup(self, stage: str) -> None:
        scaler = StandardScaler()
        dat = pd.read_csv("./diabetes.csv")
        X = torch.tensor(scaler.fit_transform(dat.iloc[:, :-1].to_numpy()), dtype=torch.float32)
        Y = torch.tensor(dat.iloc[:, -1].to_numpy(), dtype=int)
        self.dataset = TensorDataset(X, Y)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [0.6, 0.2, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

# 定义目标函数，这个函数将被Optuna调用来优化超参数
def objective(trial):
    # 超参数搜索空间
    dropout_prob = trial.suggest_float('dropout_prob', 0.5, 0.8)
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    batch_size = trial.suggest_int('batch_size', 8, 64)

    # 创建数据模块
    datamodule = LitDataModule(batch_size=batch_size)
    # trainloader, valloader, testloader = load_data(batch_size)

    # 创建模型
    model = LitModel(
        input_size=8,
        hidden_size=hidden_size,
        num_classes=2,
        dropout_prob=dropout_prob,
        lr=lr,
        weight_decay=weight_decay
    )

    # 创建 PyTorch Lightning Trainer
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir='./'),
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=300,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),  # 将 Optuna 子试验与 PyTorch Lightning callback 结合
            EarlyStopping(monitor='val_loss', patience=20, mode='min'),
            ModelCheckpoint(monitor='val_loss', mode='min')
        ]
    )

    # 训练
    #trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    #trainer.test(model, testloader, 'best')
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')

    # 获取验证集上的损失
    val_loss = trainer.callback_metrics['test_acc'].item()

    return val_loss

# 创建一个 Optuna 研究（Study），并找到最佳超参数
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
)
study.optimize(objective, n_trials=30)

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value:', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
