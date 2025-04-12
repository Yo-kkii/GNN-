import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph
import time
import psutil
import numpy as np

# 加载Cora数据集
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]


# 数据预处理
def generateSubgraphs(data, numSubgraphs=500, numHops=2):
    """从Cora图中生成多个子图模拟图分类任务"""
    subgraphs = []
    nodes = np.random.choice(data.num_nodes, numSubgraphs, replace=False)  # 随机选节点作为中心（不可重复）

    for node in nodes:
        # 提取中心节点的k-hop子图
        subset, edge_index, _, _ = k_hop_subgraph(
            node_idx=int(node),
            num_hops=numHops,
            edge_index=data.edge_index,
            relabel_nodes=True  # 标记子图节点为0,1,2...
        )

        # 创建子图Data对象
        subgraph = Data(
            x=data.x[subset],  # 子图节点特征
            edge_index=edge_index,  # 子图边连接
            y=data.y[node].unsqueeze(0)  # 标签=中心节点类别
        )
        subgraphs.append(subgraph)
    return subgraphs


# 生成子图
subgraphList = generateSubgraphs(data, numSubgraphs=500, numHops=2)


# 定义GCN模型
class GCNClassifier(nn.Module):
    def __init__(self, inputDim=dataset.num_node_features, hiddenDim=64, outputDim=dataset.num_classes):
        super().__init__()
        self.conv1 = GCNConv(inputDim, hiddenDim)
        self.conv2 = GCNConv(hiddenDim, hiddenDim)
        self.classifier = nn.Linear(hiddenDim, outputDim)
        self.dropout = nn.Dropout(0.5)



    def forward(self, x, edge_index, batch):
        # 两层GCN
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))

        # 全局平均池化
        x = global_mean_pool(x, batch)
        x=self.classifier(x)
        return x


# 训练与评估
def trainEvaluate(model, trainLoader, valLoader, testLoader):
    # Adam优化
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数
    criterion = nn.CrossEntropyLoss() 

    # 早停参数
    bestValLoss = float('inf')
    patience = 10
    patienceCounter = 0
    startTime = time.time()

# 训练循环
    for epoch in range(100):
        model.train()
        totalLoss = 0
        for batch in trainLoader:
            optimizer.zero_grad()  # 清空梯度
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()  # 优化参数
            totalLoss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {totalLoss / len(trainLoader):.4f}')

        # 验证集监控
        model.eval()
        valLoss = 0
        with torch.no_grad():
            for batch in valLoader:
                out = model(batch.x, batch.edge_index, batch.batch)
                valLoss += criterion(out, batch.y).item()
        valLoss /= len(valLoader)

        # 早停判断
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            patienceCounter = 0
        else:
            patienceCounter += 1
            if patienceCounter >= patience:
                print("早停触发")
                break
        trainTime = time.time() - startTime

# 测试评估
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in testLoader:
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)
            correct += (pred == batch.y).sum().item()
    testAccuracy = correct / len(testLoader.dataset)
    return (trainTime, testAccuracy)


def main():
    trainSize = int(0.8 * len(subgraphList))
    valSize = int(0.1 * len(subgraphList))
    testSize = int(0.1 * len(subgraphList))
    trainData, valData, testData = torch.utils.data.random_split(
        subgraphList, [trainSize, valSize, testSize]
    )

    model = GCNClassifier()
    trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
    valLoader = DataLoader(valData, batch_size=64)
    testLoader = DataLoader(testData, batch_size=64)

    trainTime, testAccuracy = trainEvaluate(model, trainLoader, valLoader, testLoader)

    # 统计输出
    print(f'\n测试准确率: {testAccuracy:.2f}')
    print(f'训练时间: {trainTime:.2f}s')
    print(f'峰值内存: {psutil.Process().memory_info().rss // 1024 ** 2}MB')


if __name__ == "__main__":
    main()
