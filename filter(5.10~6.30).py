import pandas as pd

# 1. CSV 파일 불러오기
file_path = "current_delete_videos(5.10-6.30).csv"  # 같은 폴더에 있으면 파일명만 작성
df = pd.read_csv(file_path)

# 2. 조회수 1000 이상 필터링
filtered_df = df[df["views"] >= 2000]

# 3. 결과 출력
print("조회수 1000 이상인 영상 목록:")
print(filtered_df)

# 4. 새로운 CSV 파일로 저장 (선택)
output_path = "filter(5.10~6.30).csv"
filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n결과가 '{output_path}' 파일로 저장되었습니다.")
