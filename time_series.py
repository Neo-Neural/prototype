import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. 数据生成
class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=5, output_size=2, seq_length=50):
        super(TimeSeriesDataset, self).__init__()
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.seq_length = seq_length
        self.data = []
        self.labels = []
        self.generate_data()
    
    def generate_data(self):
        for _ in range(self.num_samples):
            # 输入为随机时间序列
            input_seq = np.random.randn(self.input_size, self.seq_length)

            # 为每个输出节点生成一个线性组合
            # 生成随机权重矩阵，形状为 (output_size, input_size)
            weights = np.random.randn(self.output_size, self.input_size)

            # 应用权重到输入序列，得到输出序列
            # (output_size, input_size) @ (input_size, seq_length) = (output_size, seq_length)
            output_seq = np.dot(weights, input_seq) + 0.1 * np.random.randn(self.output_size, self.seq_length)

            self.data.append(input_seq.astype(np.float32))
            self.labels.append(output_seq.astype(np.float32))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# 2. 图结构建立
def create_graph(input_size=5, output_size=2, hidden_size=10):
    G = nx.DiGraph()
    total_nodes = input_size + hidden_size + output_size
    # 添加节点
    for i in range(total_nodes):
        if i < input_size:
            G.add_node(i, type='input')
        elif i < input_size + hidden_size:
            G.add_node(i, type='hidden')
        else:
            G.add_node(i, type='output')
    # 随机添加边，确保有向无环图或有限步循环
    # 这里为简化起见，先创建无环图
    for i in range(input_size):
        for j in range(input_size, input_size + hidden_size):
            if np.random.rand() < 0.3:
                G.add_edge(i, j, weight=np.random.randn())
    for i in range(input_size, input_size + hidden_size):
        for j in range(input_size + hidden_size, total_nodes):
            if np.random.rand() < 0.3:
                G.add_edge(i, j, weight=np.random.randn())
    # 可以选择性添加循环边或跨层边
    return G

# 3. 模型定义
class GraphNeuralNetwork(nn.Module):
    def __init__(self, graph, input_size, hidden_size, output_size, seq_length, steps=10):
        super(GraphNeuralNetwork, self).__init__()
        self.graph = graph
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_length = seq_length
        self.steps = steps  # 步骤数，用于信息传递次数
        
        self.total_nodes = input_size + hidden_size + output_size
        # 为每个节点定义一个线性变换
        self.node_layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(self.total_nodes)])
        
    def forward(self, x):
        """
        x: [batch_size, input_size, seq_length]
        """
        batch_size = x.size(0)
        device = x.device
        # 初始化节点状态
        node_states = [torch.zeros(batch_size, self.seq_length, 1).to(device) for _ in range(self.total_nodes)]
        # 设置输入节点的状态
        for i in range(self.input_size):
            node_states[i] = x[:, i, :].unsqueeze(-1)
        
        # 信息传递
        for _ in range(self.steps):
            new_states = [torch.zeros_like(state) for state in node_states]
            for src, dst in self.graph.edges():
                weight = self.graph[src][dst]['weight']
                # 将权重转换为张量，并确保与批量数据在同一设备上
                weight_tensor = torch.tensor(weight, dtype=torch.float32).to(device)
                # 信息传递: 源节点状态 * 权重
                message = node_states[src] * weight_tensor
                # 通过线性层处理传入的信息
                new_states[dst] += self.node_layers[dst](message)
            node_states = new_states  # 更新状态
        
        # 获取输出节点的状态，并去除最后一个维度
        output = torch.stack([node_states[self.input_size + self.hidden_size + i].squeeze(-1) for i in range(self.output_size)], dim=1)
        return output  # [batch_size, output_size, seq_length]

# 4. 训练过程
def train_model(model, dataloader, criterion, optimizer, num_epochs=20, device='cpu'):
    model.to(device)
    loss_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)  # [batch_size, input_size, seq_length]
            targets = targets.to(device)  # [batch_size, output_size, seq_length]
            
            optimizer.zero_grad()
            outputs = model(inputs)  # [batch_size, output_size, seq_length]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        loss_history.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    return loss_history

# 5. 可视化
def plot_loss_curve(loss_history):
    plt.figure(figsize=(10,6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

def visualize_graph(G):
    plt.figure(figsize=(12,8))
    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    for node in G.nodes(data=True):
        if node[1]['type'] == 'input':
            node_colors.append('lightgreen')
        elif node[1]['type'] == 'hidden':
            node_colors.append('skyblue')
        else:
            node_colors.append('salmon')
    nx.draw(G, pos, with_labels=True, node_color=node_colors, arrows=True)
    plt.show()

# 6. 主函数
def main():
    # 参数设置
    input_size = 5
    output_size = 2
    hidden_size = 10
    seq_length = 50
    num_samples = 1000
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    steps = 10  # 信息传递步数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    dataset = TimeSeriesDataset(num_samples=num_samples, input_size=input_size, output_size=output_size, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建图
    G = create_graph(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    visualize_graph(G)
    
    # 定义模型
    model = GraphNeuralNetwork(G, input_size, hidden_size, output_size, seq_length, steps=steps)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    loss_history = train_model(model, dataloader, criterion, optimizer, num_epochs=num_epochs, device=device)
    
    # 绘制损失曲线
    plot_loss_curve(loss_history)
    
    # 可视化优化后的图结构（可选）
    # 在此例中，图结构未经过优化，您可根据需要实现结构优化后再次可视化
    visualize_graph(G)

if __name__ == '__main__':
    main()