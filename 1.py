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
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return F.relu(self.linear(x_j))

    def update(self, aggr_out):
        return aggr_out

class RecurrentGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_time_steps):
        super(RecurrentGraphNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_time_steps = num_time_steps
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList([
            DirectedMessagePassing(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index_list):
        batch_size, num_nodes, _ = x.size()
        hidden = self.input_linear(x)

        for t in range(self.num_time_steps):
            for layer in range(self.num_layers):
                edge_index = edge_index_list[layer]
                h = self.convs[layer](hidden, edge_index)
                hidden = hidden + h

        out = self.output_linear(hidden)
        return out

def create_example_graph(num_nodes):
    edge_index = []
    for i in range(num_nodes):
        edge_index.append([i, (i + 1) % num_nodes])  # 环形连接
        edge_index.append([i, i])  # 自连接
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

# 示例参数
num_nodes = 5
input_dim = 3
hidden_dim = 16
output_dim = 2
num_layers = 2
num_time_steps = 10

edge_index_list = [create_example_graph(num_nodes) for _ in range(num_layers)]
model = RecurrentGraphNetwork(input_dim, hidden_dim, output_dim, num_layers, num_time_steps)
print(model)

# 数据准备
batch_size = 32
num_samples = 100

X = torch.randn(num_samples, num_nodes, input_dim)
Y = torch.randint(0, output_dim, (num_samples, num_nodes))
Y = F.one_hot(Y, num_classes=output_dim).float()

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_Y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X, edge_index_list)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 设置 DEAP

def evaluate(individual, edge_index_template):
    num_layers = num_layers_global
    num_nodes = num_nodes_global

    adj_vector = np.array(individual)
    adj_matrix = adj_vector.reshape(num_layers, num_nodes, num_nodes)

    new_edge_index_list = []
    for layer in range(num_layers):
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[layer, i, j] > 0.5:
                    edges.append([i, j])
        if len(edges) == 0:
            edges = [[i, i] for i in range(num_nodes)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        new_edge_index_list.append(edge_index)

    model_copy = RecurrentGraphNetwork(input_dim, hidden_dim, output_dim, num_layers, num_time_steps)
    model_copy.load_state_dict(model.state_dict())

    model_copy.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            outputs = model_copy(batch_X, new_edge_index_list)
            loss = criterion(outputs, batch_Y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return (avg_loss, )

# 全局变量
num_layers_global = num_layers
num_nodes_global = num_nodes

creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
num_edges_per_layer = num_nodes * num_nodes
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_layers * num_edges_per_layer)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate, edge_index_template=edge_index_list)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 进化过程
population = toolbox.population(n=20)
# ngen = 5
ngen = 500
cxpb = 0.5
mutpb = 0.2

for gen in range(ngen):
    print(f"=== Generation {gen+1} ===")

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    if invalid_ind:
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

    population[:] = offspring

    fits = [ind.fitness.values[0] for ind in population]
    best = population[np.argmin(fits)]
    print(f"Best Fitness: {min(fits):.4f}")