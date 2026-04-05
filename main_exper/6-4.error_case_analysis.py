"""
6-4. 오류 사례 분석
오분류 사례 추출 및 원인 분석
"""

import pandas as pd
import os


def load_predictions(kosbert_dir, gpt4o_dir):
    """
    예측 결과 로드
    
    Args:
        kosbert_dir: Ko-SBERT 결과 디렉토리
        gpt4o_dir: GPT-4o-mini 결과 디렉토리
    
    Returns:
        tuple: (kosbert_df, gpt4o_df)
    """
    
    kosbert_path = os.path.join(kosbert_dir, 'kosbert_predictions.csv')
    gpt4o_path = os.path.join(gpt4o_dir, 'gpt4o_predictions.csv')
    
    kosbert_df = pd.read_csv(kosbert_path)
    gpt4o_df = pd.read_csv(gpt4o_path)
    
    return kosbert_df, gpt4o_df


def extract_misclassified_cases(df, model_name, true_col='ground_truth', pred_col='kosbert_top1'):
    """
    오분류 사례 추출
    
    Args:
        df: 예측 결과 DataFrame
        model_name: 모델명
        true_col: 정답 컬럼명
        pred_col: 예측 컬럼명
    
    Returns:
        DataFrame: 오분류 사례
    """
    
    print(f"\n{'=' * 80}")
    print(f"{model_name} 오분류 사례 추출")
    print("=" * 80)
    
    # 오분류 필터링
    misclassified = df[df[true_col] != df[pred_col]].copy()
    
    print(f"\n총 오분류: {len(misclassified)}건 / {len(df)}건 ({len(misclassified)/len(df)*100:.1f}%)")
    
    if len(misclassified) == 0:
        print("✅ 오분류 없음 (완벽한 분류)")
        return None
    
    # 필요한 컬럼만 선택
    columns = ['post_id', '회사명', '공고제목', true_col, pred_col, 'skill_text']
    
    # 컬럼이 있는지 확인
    available_columns = [col for col in columns if col in misclassified.columns]
    misclassified_cases = misclassified[available_columns].copy()
    
    # 컬럼명 변경
    misclassified_cases = misclassified_cases.rename(columns={
        true_col: '정답_라벨',
        pred_col: '예측_라벨'
    })
    
    # skill_text 길이 추가
    if 'skill_text' in misclassified_cases.columns:
        misclassified_cases['텍스트_길이'] = misclassified_cases['skill_text'].str.len()
    
    return misclassified_cases


def analyze_error_patterns(misclassified_df, output_path):
    """
    오류 패턴 분석
    
    Args:
        misclassified_df: 오분류 사례 DataFrame
        output_path: 출력 파일 경로
    
    Returns:
        DataFrame: 오류 패턴 분석 결과
    """
    
    if misclassified_df is None or len(misclassified_df) == 0:
        return None
    
    print(f"\n📊 오류 패턴 분석")
    
    # 오류 쌍 빈도 분석
    error_pairs = misclassified_df.groupby(['정답_라벨', '예측_라벨']).size().reset_index(name='빈도')
    error_pairs = error_pairs.sort_values('빈도', ascending=False)
    
    print(f"\n가장 빈번한 오류 패턴 (상위 10개):")
    print(error_pairs.head(10).to_string(index=False))
    
    # 저장
    error_pairs.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    return error_pairs


def extract_representative_error_cases(misclassified_df, error_pairs, n_cases=10):
    """
    대표 오류 사례 추출
    
    Args:
        misclassified_df: 오분류 사례 DataFrame
        error_pairs: 오류 패턴 DataFrame
        n_cases: 추출할 사례 수
    
    Returns:
        DataFrame: 대표 오류 사례
    """
    
    if misclassified_df is None or error_pairs is None:
        return None
    
    print(f"\n📋 대표 오류 사례 추출 (상위 {n_cases}건)")
    
    representative_cases = []
    
    # 상위 오류 패턴에서 각각 1개씩 추출
    for idx, row in error_pairs.head(n_cases).iterrows():
        true_label = row['정답_라벨']
        pred_label = row['예측_라벨']
        
        # 해당 패턴의 사례 중 첫 번째 추출
        case = misclassified_df[
            (misclassified_df['정답_라벨'] == true_label) & 
            (misclassified_df['예측_라벨'] == pred_label)
        ].iloc[0]
        
        representative_cases.append(case)
    
    df_cases = pd.DataFrame(representative_cases)
    
    return df_cases


def categorize_error_types(misclassified_df):
    """
    오류 유형 분류
    
    Args:
        misclassified_df: 오분류 사례 DataFrame
    
    Returns:
        DataFrame: 오류 유형별 통계
    """
    
    if misclassified_df is None or len(misclassified_df) == 0:
        return None
    
    print(f"\n📊 오류 유형 분류")
    
    # 유사 직무 쌍 정의 (실제 분석 결과 기반으로 조정 필요)
    similar_pairs = [
        ('인공지능플랫폼구축', '인공지능서비스구현'),
        ('인공지능서비스기획', '인공지능서비스구현'),
        ('인공지능모델링', '생성형AI엔지니어링'),
        ('인공지능학습데이터구축', '인공지능모델링')
    ]
    
    def classify_error(row):
        """오류 유형 분류"""
        true_label = row['정답_라벨']
        pred_label = row['예측_라벨']
        
        # 유사 직무 오분류
        if (true_label, pred_label) in similar_pairs or (pred_label, true_label) in similar_pairs:
            return '유사직무혼동'
        
        # 텍스트 길이 기반
        if row.get('텍스트_길이', 1000) < 300:
            return '짧은텍스트'
        
        # 기타
        return '기타'
    
    if '텍스트_길이' in misclassified_df.columns:
        misclassified_df['오류유형'] = misclassified_df.apply(classify_error, axis=1)
        
        # 유형별 통계
        error_type_stats = misclassified_df['오류유형'].value_counts().reset_index()
        error_type_stats.columns = ['오류유형', '건수']
        error_type_stats['비율(%)'] = (error_type_stats['건수'] / len(misclassified_df) * 100).round(1)
        
        print(f"\n오류 유형별 분포:")
        print(error_type_stats.to_string(index=False))
        
        return error_type_stats
    
    return None


def create_error_case_report(cases_df, output_path, max_text_length=200):
    """
    오류 사례 보고서 생성
    
    Args:
        cases_df: 오류 사례 DataFrame
        output_path: 출력 파일 경로
        max_text_length: skill_text 최대 표시 길이
    """
    
    if cases_df is None or len(cases_df) == 0:
        return
    
    print(f"\n📄 오류 사례 보고서 생성 중...")
    
    # skill_text 축약
    if 'skill_text' in cases_df.columns:
        cases_df['skill_text_요약'] = cases_df['skill_text'].str[:max_text_length] + '...'
        
        # 보고서용 컬럼 선택
        report_columns = ['post_id', '회사명', '공고제목', '정답_라벨', '예측_라벨', 'skill_text_요약']
        available_columns = [col for col in report_columns if col in cases_df.columns]
        
        report_df = cases_df[available_columns].copy()
    else:
        report_df = cases_df.copy()
    
    # 저장
    report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 저장: {output_path}")
    
    # 화면 출력
    print(f"\n대표 오류 사례:")
    for idx, row in report_df.head(5).iterrows():
        print(f"\n[사례 {idx+1}]")
        print(f"  회사: {row.get('회사명', 'N/A')}")
        print(f"  공고: {row.get('공고제목', 'N/A')}")
        print(f"  정답: {row.get('정답_라벨', 'N/A')}")
        print(f"  예측: {row.get('예측_라벨', 'N/A')}")
        if 'skill_text_요약' in row:
            print(f"  내용: {row['skill_text_요약'][:100]}...")


def compare_model_errors(kosbert_misclass, gpt4o_misclass, output_path):
    """
    모델 간 오류 비교
    
    Args:
        kosbert_misclass: Ko-SBERT 오분류 사례
        gpt4o_misclass: GPT-4o-mini 오분류 사례
        output_path: 출력 파일 경로
    """
    
    if kosbert_misclass is None or gpt4o_misclass is None:
        return
    
    print(f"\n{'=' * 80}")
    print("모델 간 오류 비교")
    print("=" * 80)
    
    # post_id 기준으로 비교
    kosbert_error_ids = set(kosbert_misclass['post_id'].values)
    gpt4o_error_ids = set(gpt4o_misclass['post_id'].values)
    
    # 공통 오류
    common_errors = kosbert_error_ids & gpt4o_error_ids
    
    # 고유 오류
    kosbert_only = kosbert_error_ids - gpt4o_error_ids
    gpt4o_only = gpt4o_error_ids - kosbert_error_ids
    
    comparison_stats = {
        '구분': ['Ko-SBERT만 오분류', 'GPT-4o만 오분류', '둘 다 오분류', '전체 오분류(합집합)'],
        '건수': [len(kosbert_only), len(gpt4o_only), len(common_errors), 
                len(kosbert_error_ids | gpt4o_error_ids)]
    }
    
    df_comparison = pd.DataFrame(comparison_stats)
    
    print(f"\n")
    print(df_comparison.to_string(index=False))
    
    # 저장
    df_comparison.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    # 공통 오류 상세
    if len(common_errors) > 0:
        print(f"\n📊 두 모델 모두 오분류한 어려운 사례: {len(common_errors)}건")
        
        # 공통 오류 사례 추출
        common_df = kosbert_misclass[kosbert_misclass['post_id'].isin(common_errors)][
            ['post_id', '회사명', '공고제목', '정답_라벨', '예측_라벨']
        ].head(5)
        
        print("\n상위 5건:")
        print(common_df.to_string(index=False))


def main():
    """메인 실행"""
    
    print("=" * 80)
    print("6-4. 오류 사례 분석")
    print("=" * 80)
    
    # 디렉토리 설정
    KOSBERT_DIR = 'outputs/kosbert'
    GPT4O_DIR = 'outputs/gpt4o'
    OUTPUT_DIR = 'outputs/analysis'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 예측 결과 로드
    print("\n📂 예측 결과 로드 중...")
    kosbert_df, gpt4o_df = load_predictions(KOSBERT_DIR, GPT4O_DIR)
    print("   ✅ 로드 완료")
    
    # Ko-SBERT 오분류 사례
    kosbert_misclass = extract_misclassified_cases(
        kosbert_df, 
        'Ko-SBERT',
        pred_col='kosbert_top1'
    )
    
    if kosbert_misclass is not None:
        # 오류 패턴
        kosbert_patterns_path = os.path.join(OUTPUT_DIR, 'table_kosbert_error_patterns.csv')
        kosbert_error_pairs = analyze_error_patterns(kosbert_misclass, kosbert_patterns_path)
        
        # 대표 사례
        kosbert_cases = extract_representative_error_cases(kosbert_misclass, kosbert_error_pairs)
        kosbert_cases_path = os.path.join(OUTPUT_DIR, 'table_kosbert_error_cases.csv')
        create_error_case_report(kosbert_cases, kosbert_cases_path)
        
        # 오류 유형
        kosbert_error_types = categorize_error_types(kosbert_misclass)
        if kosbert_error_types is not None:
            kosbert_types_path = os.path.join(OUTPUT_DIR, 'table_kosbert_error_types.csv')
            kosbert_error_types.to_csv(kosbert_types_path, index=False, encoding='utf-8-sig')
    
    # GPT-4o-mini 오분류 사례
    gpt4o_misclass = extract_misclassified_cases(
        gpt4o_df, 
        'GPT-4o-mini',
        pred_col='gpt4o_top1'
    )
    
    if gpt4o_misclass is not None:
        # 오류 패턴
        gpt4o_patterns_path = os.path.join(OUTPUT_DIR, 'table_gpt4o_error_patterns.csv')
        gpt4o_error_pairs = analyze_error_patterns(gpt4o_misclass, gpt4o_patterns_path)
        
        # 대표 사례
        gpt4o_cases = extract_representative_error_cases(gpt4o_misclass, gpt4o_error_pairs)
        gpt4o_cases_path = os.path.join(OUTPUT_DIR, 'table_gpt4o_error_cases.csv')
        create_error_case_report(gpt4o_cases, gpt4o_cases_path)
        
        # 오류 유형
        gpt4o_error_types = categorize_error_types(gpt4o_misclass)
        if gpt4o_error_types is not None:
            gpt4o_types_path = os.path.join(OUTPUT_DIR, 'table_gpt4o_error_types.csv')
            gpt4o_error_types.to_csv(gpt4o_types_path, index=False, encoding='utf-8-sig')
    
    # 모델 간 오류 비교
    if kosbert_misclass is not None and gpt4o_misclass is not None:
        comparison_path = os.path.join(OUTPUT_DIR, 'table_model_error_comparison.csv')
        compare_model_errors(kosbert_misclass, gpt4o_misclass, comparison_path)
    
    print("\n" + "=" * 80)
    print("✅ 오류 사례 분석 완료")
    print("=" * 80)
    print(f"\n📁 결과 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()