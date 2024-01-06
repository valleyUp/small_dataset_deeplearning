import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(output_size, output_size)
        self.relu2 = nn.ReLU()
        
        # 捷径层，如果输入和输出尺寸不匹配，则使用这个线性层来变换
        self.shortcut = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out += identity  # 添加残差到输出
        out = self.relu2(out)
        
        return out

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_prob=0.5):
        super(ResNetBinaryClassifier, self).__init__()
        self.layers = nn.ModuleList()  # 使用ModuleList来存放各个残差块
        
        # 根据layer_sizes定义神经网络层，并且创建残差块
        current_size = input_size
        for output_size in layer_sizes:
            self.layers.append(ResidualBlock(current_size, output_size, dropout_prob))
            current_size = output_size  # 更新层的尺寸
        
        # 定义输出层
        self.output = nn.Linear(layer_sizes[-1], 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(self.output(x))  # 为二元分类应用Sigmoid
        return x
