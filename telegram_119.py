import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import requests 
import time

TELEGRAM_TOKEN = ("사용자의 텔레그램 토큰을 입력해주세요")
TELEGRAM_CHAT_ID = "(사용자의 텔레그램 아이디를 입력해주세요)"

def get_current_location():
    try:
        response = requests.get("https://ipapi.co/json/", timeout=2)
        if response.status_code == 200:
            data = response.json()
            city = data.get("city", "서울")
            region = data.get("region", "서울특별시")
            return f"{region} {city} 인근 구역"
    except Exception:
        pass
    return "등록된 지정 카메라 구역(서울)"

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"📱 [텔레그램 전송 성공] {message}")
        else:
            print(f"❌ 텔레그램 전송 실패 (코드 {response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ 텔레그램 전송 예외 발생: {e}")


ACTIONS = ["fire", "pain", "emergency", "fall", "crime"]

ACTIONS_KO = {
    "FIRE": "🔥 화재 발생",
    "PAIN": "💔 심정지 및 통증",
    "EMERGENCY": "🆘 위급 상황(도움 요청)",
    "FALL": "📉 낙상 사고 발생",
    "CRIME": "🚔 범죄 및 폭행 상황"
}

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
    print("인공지능 두뇌 및 텔레그램 시스템 연동 성공!")
else:
    print("❌ 에러: sign_model.pth 파일이 없습니다. 학습을 먼저 완료하세요.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
sequence_buffers = []

last_detected_action = None
action_start_time = None
telegram_sent = False

print("\n 텔레그램 연동 버전 실시간 테스트를 시작합니다. (종료하려면 ESC)")

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

            if confidence >= 80.0:
                if action_text == last_detected_action:
                    elapsed_time = time.time() - action_start_time
                    if elapsed_time >= 1.0 and not telegram_sent:
                        ko_status = ACTIONS_KO.get(action_text, action_text)
                        location_info = get_current_location() 

                        alert_msg = (
                            f"🚨 [119 긴급 신고 대피 시스템]\n"
                            f"📍 감지 위치: {location_info}\n"
                            f"⚠️ 상황 안내: 현재 카메라 구역에 {ko_status} 상황이 감지되었습니다!\n"
                            f"🎯 (인공지능 확신도: {confidence:.1f}%)"
                        )
                        send_telegram_message(alert_msg)
                        telegram_sent = True
                else:
                    last_detected_action = action_text
                    action_start_time = time.time()
                    telegram_sent = False
            else:
                last_detected_action = None
                telegram_sent = False

    display_text = f"PRED: {action_text} ({confidence:.1f}%)"
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    if telegram_sent:
        cv2.putText(frame, "TELEGRAM SENT!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    cv2.imshow('Real-time Sign Language Test (Telegram Active)', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
