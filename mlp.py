import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import optuna

def load_data():
    # 假设X_train, Y_train, X_val, Y_val是已经准备好的数据张量
    # X_train, Y_train是包含训练特征和标签的张量
    # X_val, Y_val是包含验证特征和标签的张量
    scaler = StandardScaler()
    dat = pd.read_csv("./diabetes.csv")
    X = torch.tensor(scaler.fit_transform(dat.iloc[:, :-2].to_numpy()), dtype=torch.float32)
    Y = torch.tensor(dat.iloc[:, -1].to_numpy(), dtype=int)
    dataset = TensorDataset(X, Y)

    # 将数据转换为PyTorch数据加载器
    train_dataset, val_dataset = random_split(dataset, [600, 168])
    batch_size = 32  # 你可以根据你的需求调整这里的批量大小
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# 定义网络结构
class SimpleDNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, dropout_prob=0.5):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 定义训练函数
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 定义评估函数
def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = correct / len(val_loader.dataset)
    return acc

# 定义目标函数，这个函数将被Optuna调用来优化超参数
def objective(trial):
    # 超参数的搜索空间
    dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.8)
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-4, log=True)
    hidden_size = trial.suggest_int('hidden_size', 128, 512)
    
    # 加载数据
    train_loader, val_loader = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 建立模型
    model = SimpleDNN(input_size=7, num_classes=2, hidden_size=hidden_size, dropout_prob=dropout_prob).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(100):
        train(model, optimizer, criterion, train_loader, device)
        
    # 评估模型
    acc = evaluate(model, val_loader, device)
    
    # Optuna希望目标函数返回一个最小化的值，所以我们返回1减去准确率
    return 1 - acc

# 创建一个Optuna研究（Study），并找到最佳超参数
study = optuna.create_study()
study.optimize(objective, n_trials=50)  # 你可以根据自己的需求调整试验的次数

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value:', 1 - trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')