import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 假设我们有一个包含每日票房的DataFrame
# 列：'movie_id', 'date', 'daily_box_office'
data = pd.read_csv('daily_box_office1.csv')

# 数据预处理
# 将日期转换为datetime格式
data['date'] = pd.to_datetime(data['date'])


# 按电影ID分组，为每部电影单独处理
def prepare_data(movie_data, look_back=7):
    # 提取每日票房数据
    daily_box_office = movie_data['daily_box_office'].values.astype('float32')

    # 标准化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    daily_box_office_scaled = scaler.fit_transform(daily_box_office.reshape(-1, 1))
    # print(daily_box_office_scaled)
    print(len(daily_box_office_scaled))

    # 创建时间序列数据集
    X, y = [], []
    for i in range(len(daily_box_office_scaled) - look_back):
        X.append(daily_box_office_scaled[i:i + look_back, 0])
        y.append(daily_box_office_scaled[i + look_back, 0])
    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


# 为每部电影准备数据
look_back = 7  # 使用过去30天的数据预测未来
all_movies_X = []
all_movies_y = []
scalers = {}  # 保存每部电影的标准化器
for movie_id, movie_data in data.groupby('movie_id'):
    X, y, scaler = prepare_data(movie_data, look_back)
    all_movies_X.append(X)
    all_movies_y.append(y)
    scalers[movie_id] = scaler  # 保存每部电影的标准化器

# 合并所有电影的数据
X = np.concatenate(all_movies_X)
y = np.concatenate(all_movies_y)

# 划分训练集和测试集
train_size = int(len(X) *1)
# print(train_size)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为PyTorch张量并移动到GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

print(len(X_train))
# 构建MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        # self.fc4 = nn.Linear(hidden_size3, output_size)
        self.fc4 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 初始化模型并移动到GPU
input_size = look_back
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16
output_size = 1
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化绘图
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
losses = []  # 用于保存每个epoch的损失

# 训练模型
num_epochs = 500
batch_size = 10
for epoch in range(num_epochs):
    epoch_loss = 0  # 用于累积每个epoch的损失
    for i in range(0, len(X_train), batch_size):
        # 获取小批量数据
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 累积损失

    # 计算每个epoch的平均损失
    # print(len(X_train))
    epoch_loss /= (len(X_train) // batch_size)
    losses.append(epoch_loss)

    # 实时绘制损失曲线
    ax.clear()
    ax.plot(losses, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    plt.pause(0.1)  # 短暂暂停以更新图表

    # 打印每个epoch的平均损失
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 关闭交互模式
plt.ioff()
plt.show()
print("xxx")
for movie_id, movie_data in data.groupby('movie_id'):
    if movie_id == 1:
        X, y, scaler = prepare_data(movie_data, look_back)
        all_movies_X.append(X)
        all_movies_y.append(y)
        scalers[movie_id] = scaler  # 保存每部电影的标准化器

# 合并所有电影的数据
X = np.concatenate(all_movies_X)
y = np.concatenate(all_movies_y)

# 划分训练集和测试集
train_size = 0
# print(train_size)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为PyTorch张量并移动到GPU
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
# 在测试集上验证模型
with torch.no_grad():
    print("aaa")
    y_pred = model(X_test).cpu().numpy()  # 将结果移回CPU
    y_test = y_test.cpu().numpy()
print("bbb")
# 反标准化预测结果
y_pred_actual = scalers[1].inverse_transform(y_pred)  # 假设使用第一部电影的标准化器
y_test_actual = scalers[1].inverse_transform(y_test)

# 计算均方误差
mse = mean_squared_error(y_test_actual, y_pred_actual)
print(f'Test MSE: {mse}')

# 输出测试集的前10个预测结果和实际结果
print("\nTest Set Predictions vs Actual:")
for i in range(10):
    print(f"Predicted: {y_pred_actual[i][0]:.2f}, Actual: {y_test_actual[i][0]:.2f}")

# 保存模型和标准化器
torch.save(model.state_dict(), 'box_office_model_3_500.pth')
import pickle

with open('scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)

print("模型和标准化器已保存！")