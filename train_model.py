import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

current_folder = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_folder, 'data_all.csv')

if not os.path.exists(data_path):
    print("에러: data_all.csv 파일이 없습니다. 먼저 data_merge.py를 실행하세요.")
    exit()

data = pd.read_csv(data_path, header=None).values
X = data[:, :-1] 
y = data[:, -1]   

X = X.reshape(-1, 40, 126)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

class SignLanguageLSTM(nn.Module):
    def __init__(self):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=126, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 5)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

model = SignLanguageLSTM()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

print("인공지능 모델 학습을 시작합니다")

epochs = 150

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0) * 100
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | 검증 정확도: {accuracy:.1f}%")

model_path = os.path.join(current_folder, 'sign_model.pth')
torch.save(model.state_dict(), model_path)

print(f"\n==================================================")
print(f"학습 완료 두뇌 파일 업데이트됨: sign_model.pth")
print(f"==================================================")