import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. 数据生成
class StaticDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=5, output_size=2):
        super(StaticDataset, self).__init__()
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.data = []
        self.labels = []
        self.generate_data()
    
    def generate_data(self):
        noise_rate = 0
        for _ in range(self.num_samples):
            # 输入为随机向量
            input_vector = np.random.randn(self.input_size)
            # 输出为输入的某种组合，例如线性变换，加入噪声
            weight_matrix = np.random.randn(self.input_size, self.output_size)
            output_vector = input_vector @ weight_matrix + noise_rate * np.random.randn(self.output_size)
            self.data.append(input_vector.astype(np.float32))
            self.labels.append(output_vector.astype(np.float32))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# 2. 图结构建立
def create_graph(input_size=5, output_size=2, hidden_size=10, edge_prob=0.3):
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
    # 添加边：输入 -> 隐藏，隐藏 -> 隐藏，隐藏 -> 输出
    for i in range(input_size):
        for j in range(input_size, input_size + hidden_size):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j, weight=np.random.randn())
    for i in range(input_size, input_size + hidden_size):
        for j in range(input_size, input_size + hidden_size):
            if i != j and np.random.rand() < edge_prob:
                G.add_edge(i, j, weight=np.random.randn())
    for i in range(input_size, input_size + hidden_size):
        for j in range(input_size + hidden_size, total_nodes):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j, weight=np.random.randn())
    return G

# 3. 模型定义
class GraphNeuralNetwork(nn.Module):
    def __init__(self, graph, input_size, hidden_size, output_size, steps=3):
        super(GraphNeuralNetwork, self).__init__()
        self.graph = graph
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.steps = steps  # 信息传递步数
        
        self.total_nodes = input_size + hidden_size + output_size
        # 定义节点的特征变换层
        self.node_features = nn.Parameter(torch.randn(self.total_nodes, 16))  # 初始特征维度为16
        self.message_passing = nn.ModuleList([
            nn.Linear(16, 16) for _ in range(self.steps)
        ])
        self.output_layer = nn.Linear(16, 1)  # 最终输出层
        
    def forward(self, x):
        """
        x: [batch_size, input_size]
        """
        batch_size = x.size(0)
        device = x.device

        # 初始化节点特征
        node_states = self.node_features.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, total_nodes, 16]
        # 设置输入节点的特征为输入向量的线性变换
        input_transform = nn.Linear(self.input_size, 16).to(device)
        node_states[:, :self.input_size, :] = input_transform(x).unsqueeze(1).repeat(1, self.input_size, 1)
        
        # 信息传递
        for step in range(self.steps):
            messages = torch.zeros_like(node_states)
            for src, dst in self.graph.edges():
                weight = self.graph[src][dst]['weight']
                weight_tensor = torch.tensor(weight, dtype=torch.float32).to(device)
                messages[:, dst, :] += self.message_passing[step](node_states[:, src, :]) * weight_tensor
            node_states = messages + node_states  # 叠加当前状态
        
        # 获取输出节点的特征
        output_nodes = node_states[:, self.input_size + self.hidden_size:, :]  # [batch_size, output_size, 16]
        # 通过输出层得到最终输出
        output = self.output_layer(output_nodes).squeeze(-1)  # [batch_size, output_size]
        return output

# 4. 训练过程
def train_model(model, dataloader, criterion, optimizer, num_epochs=20, device='cpu'):
    model.to(device)
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)   # [batch_size, input_size]
            targets = targets.to(device) # [batch_size, output_size]
            
            optimizer.zero_grad()
            outputs = model(inputs)      # [batch_size, output_size]
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
    nx.draw(G, pos, with_labels=True, node_color=node_colors, arrows=True, node_size=500)
    plt.show()

# 6. 主函数
def main():
    # 参数设置
    input_size = 5
    output_size = 2
    hidden_size = 10
    num_samples = 1000
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    steps = 3  # 信息传递步数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    dataset = StaticDataset(num_samples=num_samples, input_size=input_size, output_size=output_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建图
    G = create_graph(input_size=input_size, output_size=output_size, hidden_size=hidden_size, edge_prob=0.3)
    visualize_graph(G)
    
    # 定义模型
    model = GraphNeuralNetwork(G, input_size, hidden_size, output_size, steps=steps)
    
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