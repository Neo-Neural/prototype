
# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
torch.manual_seed(0)
np.random.seed(0)

# 生成模拟的时间序列数据
def generate_time_series(seq_length):
    """
    生成一个简单的正弦波作为时间序列数据，并添加一些噪声。
    """
    x = np.linspace(0, 100, seq_length)
    y = np.sin(x) + 0.1 * np.random.randn(seq_length)
    
    # another y
    y = np.log10(x + 10) - 2 + 0.02 * np.random.randn(seq_length)
    return y

# 定义逻辑规则
def logical_rules(pred, target):
    """
    定义一些简单的逻辑规则，用于约束预测结果。
    例如，预测值应该在合理的范围内。
    """
    # 示例规则：预测值不能超过1.5或低于-1.5
    rule1 = pred < 1.5
    rule2 = pred > -1.5
    return rule1 & rule2

# 定义逻辑神经网络模型
class LogicalNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化逻辑神经网络。
        """
        super(LogicalNeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        前向传播函数。
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out

# 准备数据
seq_length = 200
data = generate_time_series(seq_length)

# 可视化原始数据
plt.figure(figsize=(10,4))
plt.plot(data, label='Original Data')
plt.legend()
plt.show()

# 数据预处理：创建输入和目标
def create_inout_sequences(data, seq_length):
    """
    将时间序列数据转换为适合模型训练的输入和目标。
    """
    inout_seq = []
    L = len(data)
    for i in range(L - seq_length):
        train_seq = data[i:i+seq_length]
        train_label = data[i+seq_length]
        inout_seq.append((train_seq, train_label))
    return inout_seq

sequence_length = 20  # 使用前20个时间步预测下一个时间步
inout_seq = create_inout_sequences(data, sequence_length)

# 将数据转换为张量
train_data = torch.FloatTensor([item[0] for item in inout_seq]).unsqueeze(2)  # 形状: [样本数, 序列长度, 特征数]
train_labels = torch.FloatTensor([item[1] for item in inout_seq]).unsqueeze(1)  # 形状: [样本数, 1]

# 初始化模型、损失函数和优化器
input_size = 1      # 每个时间步一个特征
hidden_size = 50    # LSTM的隐藏层大小
output_size = 1     # 输出一个值
model = LogicalNeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    
    # 前向传播
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    
    # 应用逻辑规则（移除 torch.no_grad() 块）
    predictions = outputs
    rules = logical_rules(predictions, train_labels)
    # 如果预测不满足规则，则增加一个大的惩罚
    rule_penalty = torch.where(rules, torch.zeros_like(loss), torch.ones_like(loss) * 10)
    total_loss = loss + rule_penalty.mean()
    
    # 反向传播和优化
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # 每隔一定步数输出一次损失
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_outputs = model(train_data).squeeze().numpy()

# 可视化预测结果
plt.figure(figsize=(10,4))
plt.plot(data, label='Original Data')
plt.plot(range(sequence_length, seq_length), test_outputs, label='Predicted Data')
plt.legend()
plt.show()