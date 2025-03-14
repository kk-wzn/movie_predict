import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt


# 加载模型
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


# 初始化模型
input_size = 7  # 与训练时一致
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16
output_size = 1
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# 加载模型参数
model.load_state_dict(torch.load('box_office_model_3_500.pth'))
model.eval()  # 设置为评估模式

# 加载标准化器
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)


# 预测
def predict_future(model, initial_data, scaler, future_days=1):
    predictions = []
    current_batch = initial_data.reshape((1, input_size))
    for _ in range(future_days):

        current_batch_tensor = torch.tensor(current_batch, dtype=torch.float32)


        with torch.no_grad():
            current_pred = model(current_batch_tensor).numpy()[0][0]
        predictions.append(current_pred)


        current_batch = np.append(current_batch[:, 1:], [[current_pred]], axis=1)


    predictions_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions_actual



test_data = pd.read_csv('daily_box_office_test.csv')
test_data['date'] = pd.to_datetime(test_data['date'])

test_movie_id = 1
test_movie_data = test_data[test_data['movie_id'] == test_movie_id]
test_movie_daily_box_office = test_movie_data['daily_box_office'].values.astype('float32')

print(test_movie_daily_box_office)
# norm
scaler = scalers[test_movie_id]
test_movie_daily_box_office_scaled = scaler.transform(test_movie_daily_box_office.reshape(-1, 1))

ans =[]
begindate =0
for i in range(input_size+begindate):
    ans.append(0)
print(ans)

for i in range(begindate,37):
    initial_data = test_movie_daily_box_office_scaled[i:i+input_size:]
    # print(initial_data)
    future_predictions = predict_future(model, initial_data, scaler, future_days=1)
    ans.append(future_predictions[0][0])
    # print(f'Future Daily Box Office Predictions: {future_predictions[0][0]}')
    pass
print(ans)
fig = plt.figure(dpi=128, figsize=(10, 6))
plt.plot(ans)
plt.plot(test_movie_daily_box_office)
plt.show()
