import torch.nn as nn
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

# 构建边
# 这需要您根据视频的内容和所需的图结构来定义
def build_edges_based_on_criteria(video,keyframe_index_list,gt_summ,dataset=None,sim_threshold=1):
    batch_size = len(keyframe_index_list)
    most_similar_frame_feature = []
    test = []
    gat_model = GAT(128,64,128,64,4)
    for i in range(batch_size):
        all_test_lists = []
        if dataset == 'Daily_Mail':
            test = []
            pred_summ = video[i][keyframe_index_list[i]]
            pred_summ = F.normalize(pred_summ, dim=1)
            test.append(pred_summ)
            # 前十帧
            for j in range(keyframe_index_list[i] - 10, keyframe_index_list[i]-1):
                if 0 <= j < len(video[i]):
                    frame = video[i][j]
                    frame = F.normalize(frame, dim=1)
                    test.append(frame)
            # 后十帧
            for j in range(keyframe_index_list[i] + 1, keyframe_index_list[i] + 10):
                if 0 <= j < len(video[i]):
                    frame = video[i][j]
                    frame = F.normalize(frame, dim=1)
                    test.append(frame)
            all_test_lists.append(test)
        elif dataset == 'BLiSS':
            test = []
            pred_summ = video[i]
            pred_summ = F.normalize(pred_summ, dim=1)
            test.append(pred_summ)
            # 前十帧
            for j in range(keyframe_index_list[i] - 10, keyframe_index_list[i]-1):
                if 0 <= j < len(video[i]):
                    frame = video[i][j]
                    frame = F.normalize(frame, dim=1)
                    test.append(frame)
            # 后十帧
            for j in range(keyframe_index_list[i] + 1, keyframe_index_list[i] + 10):
                if 0 <= j < len(video[i]):
                    frame = video[i][j]
                    frame = F.normalize(frame, dim=1)
                    test.append(frame)
            all_test_lists.append(test)
        for i, test in all_test_lists:
            node_features = []  # 用于存储节点特征
            edges = []  # 用于存储边的信息
            for j, frames in test:
                node_features.append(frames)  # 将帧特征添加到节点特征列表中
                if j > 0:
                    # 计算当前帧与上一帧的相似性，假设使用相似性矩阵或者其他方法计算相似性
                    similarity = similarities(test[j - 1], test[j])
                    if similarity > sim_threshold:
                        edges.append((j - 1, j))
            # 构建图的邻接矩阵
            edge_index = torch.tensor([[x[0], x[1]] for x in edges], dtype=torch.long).t()

            # 将节点特征转换为张量
            x = torch.tensor(node_features, dtype=torch.float)

            # 构建 PyTorch Geometric 的 Data 对象
            data = Data(x=x, edge_index=edge_index)

            # 将数据传递给 GAT 模型进行处理
            output = gat_model(data)

            # 计算输出与目标之间的相似性分数（余弦相似度）
            similarity_scores = F.cosine_similarity(output, pred_summ, dim=1)

            # 找到前十帧中相似性最高的两帧特征的索引
            most_similar_index_front = torch.argsort(similarity_scores[1:11])[:2] + 1
            most_similar_frame_feature.append(keyframe_index_list[i] - 10 + most_similar_index_front[0])
            most_similar_frame_feature.append(keyframe_index_list[i] - 10 + most_similar_index_front[1])
            most_similar_frame_feature.append(keyframe_index_list[i] - 10 + most_similar_index_front[2])
            most_similar_frame_feature.append(keyframe_index_list[i] - 10 + most_similar_index_front[3])
            # 找到后十帧中相似性最高的两帧特征的索引
            most_similar_index_back = torch.argsort(similarity_scores[11:])[:2] + 11
            most_similar_frame_feature.append(keyframe_index_list[i] + 10 - most_similar_index_back[0])
            most_similar_frame_feature.append(keyframe_index_list[i] + 10 - most_similar_index_back[1])
            most_similar_frame_feature.append(keyframe_index_list[i] + 10 - most_similar_index_back[2])
            most_similar_frame_feature.append(keyframe_index_list[i] + 10 - most_similar_index_back[3])

            ''
            # # 最相似的帧特征
            # most_similar_frame_feature_front = [node_features[index] for index in most_similar_index_front]
            # most_similar_frame_feature_back = [node_features[index] for index in most_similar_index_back]
            # most_similar_frame_feature.extend((most_similar_frame_feature_front, most_similar_frame_feature_back))


    return most_similar_frame_feature
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
def build_edges(x, sim_threshold):
    edges = []
    num_frames = x.size(0)
    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            similarity = similarities(x[i], x[j])
            if similarity > sim_threshold:
                edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads))
        self.lin = nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def similarities(preframe,nextframe):
 # 计算前后帧之间的相似性
    similarities = np.dot(preframe, nextframe.T)
    norms = np.linalg.norm(preframe, axis=1, keepdims=True) * np.linalg.norm(nextframe,axis=1, keepdims=True).T
    similarities /= norms