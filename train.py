import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_dense_adj, dense_to_sparse
import torch_geometric.transforms as T

import numpy as np
import random

# 用于进化算法的库
from deap import base, creator, tools, algorithms

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DirectedMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DirectedMessagePassing, self).__init__(aggr='add')  # 选择聚合方式
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: 节点特征矩阵
        # edge_index: [2, num_edges]
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j: 接收节点的特征
        return F.relu(self.linear(x_j))

    def update(self, aggr_out):
        # 更新节点特征
        return aggr_out
      
class RecurrentGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_time_steps):
        super(RecurrentGraphNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_time_steps = num_time_steps
        self.hidden_dim = hidden_dim

        # 定义多层图神经网络
        self.convs = nn.ModuleList([
            DirectedMessagePassing(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 输入层
        self.input_linear = nn.Linear(input_dim, hidden_dim)

        # 输出层
        self.output_linear = nn.Linear(hidden_dim, output_dim)

        # 图结构参数
        # 使用可学习的邻接矩阵（通过Sigmoid限制在0-1之间）
        self.adj_matrix = nn.Parameter(torch.rand(num_layers, hidden_dim, hidden_dim))

    def forward(self, x, edge_index_list):
        """
        x: [batch_size, num_nodes, input_dim]
        edge_index_list: List of edge_index tensors for each layer
        """
        batch_size, num_nodes, _ = x.size()

        # 初始化隐藏状态
        hidden = self.input_linear(x)  # [batch_size, num_nodes, hidden_dim]

        for t in range(self.num_time_steps):
            for layer in range(self.num_layers):
                edge_index = edge_index_list[layer]  # 获取当前层的边索引
                h = hidden[:, :, :]  # 当前隐藏状态
                h = self.convs[layer](h, edge_index)
                hidden = hidden + h  # 简单的残差连接

        out = self.output_linear(hidden)  # [batch_size, num_nodes, output_dim]
        return out
      
def create_example_graph(num_nodes):
    """
    创建一个包含环的有向图示例
    """
    # 创建环形连接
    edge_index = []
    for i in range(num_nodes):
        edge_index.append([i, (i + 1) % num_nodes])  # 将每个节点连接到下一个节点，形成环
        # 添加自连接
        edge_index.append([i, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

# 示例参数
num_nodes = 5
input_dim = 3
hidden_dim = 16
output_dim = 2
num_layers = 2
num_time_steps = 10

# 创建多个层的边索引（这里简单复制同一个环形连接）
edge_index_list = [create_example_graph(num_nodes) for _ in range(num_layers)]

# 创建模型实例
model = RecurrentGraphNetwork(input_dim, hidden_dim, output_dim, num_layers, num_time_steps)

# 打印模型结构
print(model)

# 生成随机输入和目标输出
batch_size = 32
num_samples = 100

# 随机输入数据
X = torch.randn(num_samples, num_nodes, input_dim)

# 随机目标（例如分类任务）
Y = torch.randint(0, output_dim, (num_samples, num_nodes))
Y = F.one_hot(Y, num_classes=output_dim).float()

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X, edge_index_list)  # [batch_size, num_nodes, output_dim]
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")