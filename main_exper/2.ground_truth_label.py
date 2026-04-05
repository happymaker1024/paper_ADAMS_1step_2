"""
Ground Truth 라벨링 템플릿 생성
2인 독립 라벨링을 위한 Excel 파일 생성
"""

import pandas as pd
import os


def create_labeling_template(input_csv, output_excel):
    """
    Ground Truth 라벨링 템플릿 생성
    
    Args:
        input_csv: 전처리된 데이터 CSV 경로
        output_excel: 출력할 Excel 파일 경로
    """
    
    print("=" * 80)
    print("Ground Truth 라벨링 템플릿 생성")
    print("=" * 80)
    
    # 데이터 로드
    print(f"\n📂 데이터 로드: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   총 {len(df)}건")
    
    # 라벨링용 데이터 생성
    labeling_df = df[['post_id', '회사명', '공고제목', '직무명', '직무명1', '직무명2', 'skill_text']].copy()
    
    # 라벨링 컬럼 추가
    labeling_df['평가자1_1순위'] = ''
    labeling_df['평가자1_후보1'] = ''
    labeling_df['평가자1_후보2'] = ''
    labeling_df['평가자2_1순위'] = ''
    labeling_df['평가자2_후보1'] = ''
    labeling_df['평가자2_후보2'] = ''
    labeling_df['최종합의_라벨'] = ''
    labeling_df['메모'] = ''
    
    # Excel로 저장
    print(f"\n💾 Excel 파일 생성 중...")
    
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # 라벨링 시트
        labeling_df.to_excel(writer, sheet_name='라벨링', index=False)
        
        # 가이드 시트
        guide_data = {
            '항목': [
                'NCS 세분류 7개',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '라벨링 방법',
                '',
                '',
                '',
                '판단 기준',
                '',
                '',
                '',
            ],
            '내용': [
                '1. 인공지능플랫폼구축',
                '2. 인공지능서비스기획',
                '3. 인공지능모델링',
                '4. 인공지능서비스운영관리',
                '5. 인공지능서비스구현',
                '6. 인공지능학습데이터구축',
                '7. 생성형AI엔지니어링',
                '',
                '1. skill_text 전체를 읽고 판단',
                '2. 1순위 세분류 1개 선택 (필수)',
                '3. 후보 세분류 최대 2개 선택 (선택)',
                '4. 직무명이 아닌 업무 내용 중심으로 판단',
                '',
                '• 주요업무, 자격요건, 우대사항에서 가장 핵심적인 업무 비중',
                '• 모델 설계/학습 중심 → 인공지능모델링',
                '• 서비스 기획/요구분석 중심 → 인공지능서비스기획',
            ]
        }
        guide_df = pd.DataFrame(guide_data)
        guide_df.to_excel(writer, sheet_name='라벨링가이드', index=False)
    
    print(f"   ✅ 저장 완료: {output_excel}")
    print(f"\n📊 생성된 컬럼:")
    print(f"   - post_id: 공고 ID")
    print(f"   - 회사명, 공고제목, 직무명, skill_text")
    print(f"   - 평가자1_1순위, 평가자1_후보1, 평가자1_후보2")
    print(f"   - 평가자2_1순위, 평가자2_후보1, 평가자2_후보2")
    print(f"   - 최종합의_라벨, 메모")
    
    print(f"\n📋 직무별 분포:")
    job_counts = df['직무명'].value_counts().sort_index()
    for job, count in job_counts.items():
        print(f"   {job}: {count}건")
    
    print("\n" + "=" * 80)
    print("✅ 라벨링 템플릿 생성 완료")
    print("=" * 80)
    print("\n다음 단계:")
    print("1. Excel 파일을 평가자 2명에게 배포")
    print("2. 각자 독립적으로 라벨링 수행")
    print("3. 결과 수합 후 Cohen's Kappa 계산")
    print("4. 불일치 항목 논의 및 최종 합의")


if __name__ == "__main__":
    # 파일 경로
    INPUT_CSV = 'datas/preprocessed/preprocessed_jobposting_with_postid.csv'
    OUTPUT_EXCEL = 'datas/ground_truth/ground_truth_labeling_template.xlsx'
    
    # 템플릿 생성
    create_labeling_template(INPUT_CSV, OUTPUT_EXCEL)