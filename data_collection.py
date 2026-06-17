import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

ACTION_ID = 0  # 0: 화재, 1: 통증, 2: 위급, 3: 낙상, 4: 범죄
ACTION_NAME = "fire"  # 파일 이름에 들어갈 영문명

TOTAL_SEQUENCES = 50  # 총 촬영 횟수 (50번 반복)
SEQUENCE_LENGTH = 40  # 한 번 촬영할 때 수집할 프레임 수 (30프레임 = 약 1초)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("에러: 웹캠을 켜지 못했습니다. 카메라 연결을 확인하세요.")
    exit()

dataset = []

print(f" [{ACTION_NAME.upper()}] 수어 데이터 수집을 준비합니다. 3초 뒤 시작...")
time.sleep(3)

for seq in range(TOTAL_SEQUENCES):
    sequence_data = []
    print(f" {seq + 1}/{TOTAL_SEQUENCES} 번째 촬영 시작!")
    
    while len(sequence_data) < SEQUENCE_LENGTH:
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
                
        sequence_data.append(joint_array)
        
        status_text = f"Action: {ACTION_NAME} | Seq: {seq+1}/{TOTAL_SEQUENCES} | Frames: {len(sequence_data)}/{SEQUENCE_LENGTH}"
        cv2.putText(frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Sign Language Data Collection', frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC 누르면 종료
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    sequence_data = np.array(sequence_data)
    flatten_data = sequence_data.flatten()
    row = np.append(flatten_data, ACTION_ID)
    dataset.append(row)
    time.sleep(1)

file_name = f"data_{ACTION_NAME}.csv"
df = pd.DataFrame(dataset)
df.to_csv(file_name, index=False, header=False)

cap.release()
cv2.destroyAllWindows()
print(f" [{ACTION_NAME.upper()}] 수집 완료! 파일 저장됨: {file_name}")