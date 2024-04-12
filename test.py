import torch
import torch.nn.functional as F

# 假设我们有一个batch_size为2的数据
# predictions是模型的预测输出，形状为[batch_size, num_classes]
predictions = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.2, 0.0]])

# targets是真实类别索引，形状为[batch_size]
targets = torch.tensor([1, 0])

# 计算交叉熵损失
loss = F.cross_entropy(predictions, targets)

print(loss)