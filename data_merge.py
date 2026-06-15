import pandas as pd
import glob
import os

# 현재 폴더에 있는 모든 data_*.csv 파일 찾기
current_folder = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(current_folder, 'data_*.csv'))

all_data = []
for file in csv_files:
    # 혹시 기존에 합쳐둔 파일이 있다면 제외하고 순수 데이터만 가져오기
    if 'data_all.csv' in file:
        continue
    df = pd.read_csv(file, header=None)
    all_data.append(df)
    print(f" {os.path.basename(file)} 통합 중... (데이터 개수: {len(df)}개)")

# 하나의 데이터로 합치기
total_df = pd.concat(all_data, axis=0, ignore_index=True)

# data_all.csv로 저장
output_path = os.path.join(current_folder, 'data_all.csv')
total_df.to_csv(output_path, index=False, header=False)

print(f"==================================================")
print(f" 총 {len(total_df)}개의 데이터 세트 통합 완료! -> data_all.csv 생성됨")
print(f"==================================================")