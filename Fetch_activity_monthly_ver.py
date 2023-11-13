import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
from torch import nn
import os
import calendar

data = pd.read_csv("data_daily.csv")
base_date = datetime.datetime.strptime(data.iloc[0,0], "%Y-%m-%d").date()
date_list = [datetime.datetime.strptime(i,"%Y-%m-%d").date() for i in data.iloc[:,0].tolist()]
data_list = data.iloc[:,1].tolist()
month_dict = {}
for i in range(len(date_list)):
  d = date_list[i]
  if d.month not in month_dict:
    month_dict[d.month] = {"days_count": 0,
                           "receipts_count": 0}
  month_dict[d.month]["days_count"] += 1
  month_dict[d.month]["receipts_count"] += data_list[i]

x_list = []
y_list = []
for i in month_dict:
  month_dict[i]["avg_receipt"] = month_dict[i]["receipts_count"] / month_dict[i]["days_count"]
  x_list.append(i)
  y_list.append(month_dict[i]["avg_receipt"])

x = torch.tensor(x_list).unsqueeze(dim=1)
y = torch.tensor(y_list).unsqueeze(dim=1)

train_length = int(len(x)*0.8)
x_train = x[:train_length]
x_test = x[train_length:]
y_train = y[:train_length]
y_test = y[train_length:]

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
torch.manual_seed(9320)
model_0 = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model_0(x_test)

loss_ = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 2200)

epochs = 10000


train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()

    with torch.inference_mode():
      test_pred = model_0(x_test)
      test_loss = loss_(test_pred, y_test.type(torch.float))
      if epoch % 1000 == 0:
            print("ML processing...")

def generate_prediction(userInput):
      input_date = datetime.datetime.strptime(userInput, "%Y-%m").date()
      lst = list(model_0.parameters())
      monthdiff = (input_date.year - base_date.year)*12+input_date.month-base_date.month
      days_in_input_month = calendar.monthrange(input_date.year, input_date.month)[1]
      weight = lst[0].item()
      bias = lst[1].item()
      daily_estimation = int(monthdiff) * weight + bias
      monthly_estimation = daily_estimation * days_in_input_month
      return [daily_estimation, monthly_estimation]

def plot_predictions(train_data=x_train, 
                     train_labels=y_train, 
                     test_data=x_test, 
                     test_labels=y_test):
  lst = list(model_0.parameters())
  weight = lst[0].item()
  bias = lst[1].item()
  x_prediction = [i+12 for i in x_list]
  y_regression = [i * weight + bias for i in x_prediction]
  
  plt.figure(figsize=(10, 7))

  plt.scatter(train_data, train_labels, c="b", s=10, label="Training data")
  
  plt.scatter(test_data, test_labels, c="g", s=10, label="Testing data")

  plt.scatter(x_prediction, y_regression, c='r', label = 'Prediction')