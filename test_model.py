import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os

ACTIONS = ["fire", "pain", "emergency", "fall", "crime"]

class SignLanguageLSTM(nn.Module):
    def __init__(self):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=126, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 5)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

current_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_folder, 'sign_model.pth')

model = SignLanguageLSTM()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("인공지능 두뇌(sign_model.pth) 로드 성공")
else:
    print("에러: sign_model.pth 파일이 없습니다. 2번 코드를 먼저 실행해서 모델을 만드세요.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
sequence_buffers = [] 

print("\n실시간 수어 인식 테스트를 시작합니다")
print(" 카메라를 보며 수어 동작을 취해보세요 (종료하려면 ESC)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    joint_array = np.zeros(126)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            hand_joints = []
            for lm in hand_landmarks.landmark:
                hand_joints.extend([lm.x, lm.y, lm.z])
            joint_array[i*63 : (i+1)*63] = hand_joints
            
    sequence_buffers.append(joint_array)
    
    if len(sequence_buffers) > 40:
        sequence_buffers.pop(0)
        
    action_text = "Waiting..."
    confidence = 0.0
    
    if len(sequence_buffers) == 40 and len(sequence_buffers) % 5 == 0:
        test_input = np.array(sequence_buffers)
        test_input = torch.FloatTensor(test_input).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(test_input)
            probabilities = torch.softmax(outputs, dim=1)[0]
            idx = torch.argmax(probabilities).item()
            
            action_text = ACTIONS[idx].upper()
            confidence = probabilities[idx].item() * 100

    display_text = f"PRED: {action_text} ({confidence:.1f}%)"
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow('Real-time Sign Language Test (40 Frames)', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()