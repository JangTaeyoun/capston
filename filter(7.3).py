import pandas as pd

# 입력 파일명
filtered_node_file = 'filter(5.10~6.30).csv'
current_file = 'current_videos(7.3).csv'
output_file = 'filter(7.3).csv'

# 필터링된 노드(영상) 목록 읽기
try:
    filtered_df = pd.read_csv(filtered_node_file, encoding='utf-8-sig')
except Exception:
    filtered_df = pd.read_csv(filtered_node_file)

# 현재 영상 데이터 읽기
try:
    current_df = pd.read_csv(current_file, encoding='utf-8-sig')
except Exception:
    current_df = pd.read_csv(current_file)

# 'video_id' 컬럼이 없는 경우 에러 처리
if 'video_id' not in filtered_df.columns or 'video_id' not in current_df.columns:
    raise ValueError("'video_id' 컬럼이 없습니다. 파일을 확인하세요.")

# delete_node(5.10-6.30).csv에 있는 영상만 current_videos(7.3).csv에서 추출
filtered_video_ids = set(filtered_df['video_id'])
matched_df = current_df[current_df['video_id'].isin(filtered_video_ids)].reset_index(drop=True)

# 결과 저장
matched_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"filter(5.10-6.30).csv에 있는 영상만 추출하여 저장 완료: {output_file}")
print(f"원본 영상 수: {len(current_df)}, 필터링 후 영상 수: {len(matched_df)}") 
