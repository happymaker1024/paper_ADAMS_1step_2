"""
채용공고 데이터 전처리 파이프라인
논문 III.4절 "데이터 전처리" 구현

프로세스:
1. 원본 데이터 로드 (307건)
2. 중복 제거 (284건)
3. 텍스트 정제 및 skill_text 생성
4. 짧은 텍스트 필터링 (282건)
"""

import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


def step1_load_data(input_path):
    """
    Step 1: 원본 데이터 로드
    
    Args:
        input_path: 입력 CSV 파일 경로
    
    Returns:
        DataFrame: 로드된 원본 데이터
    """
    print("\n" + "=" * 80)
    print("STEP 1: 원본 데이터 로드")
    print("=" * 80)
    
    df = pd.read_csv(input_path)
    
    print(f"✅ 파일 로드 완료: {input_path}")
    print(f"   총 {len(df)}건, {len(df.columns)}개 컬럼")
    print(f"\n📊 컬럼 목록:")
    for col in df.columns:
        print(f"   - {col}")
    
    print(f"\n📊 직무별 분포 (원본):")
    job_counts = df['직무명'].value_counts().sort_index()
    for job, count in job_counts.items():
        print(f"   {job}: {count}건")
    print(f"   합계: {len(df)}건")
    
    return df


def step2_remove_duplicates(df):
    """
    Step 2: 중복 제거
    
    중복 기준: 회사명 + 공고제목 + 주요업무
    
    Args:
        df: 입력 DataFrame
    
    Returns:
        DataFrame: 중복 제거된 데이터
    """
    print("\n" + "=" * 80)
    print("STEP 2: 중복 제거")
    print("=" * 80)
    
    original_count = len(df)
    print(f"중복 제거 전: {original_count}건")
    
    # 중복 제거 (첫 번째 항목 유지)
    df_dedup = df.drop_duplicates(
        subset=['회사명', '공고제목', '주요업무'], 
        keep='first'
    )
    
    dedup_count = len(df_dedup)
    removed_count = original_count - dedup_count
    
    print(f"중복 제거 후: {dedup_count}건")
    print(f"제거된 중복: {removed_count}건")
    
    if removed_count > 0:
        # 제거된 항목 샘플 출력
        duplicated_mask = df.duplicated(
            subset=['회사명', '공고제목', '주요업무'], 
            keep='first'
        )
        duplicates = df[duplicated_mask]
        
        print(f"\n📋 제거된 중복 샘플 (상위 5건):")
        for idx, row in duplicates.head(5).iterrows():
            print(f"   [{row['직무명']}] {row['회사명']} - {row['공고제목'][:40]}...")
    
    print(f"\n📊 직무별 분포 (중복 제거 후):")
    job_counts = df_dedup['직무명'].value_counts().sort_index()
    for job, count in job_counts.items():
        print(f"   {job}: {count}건")
    
    return df_dedup.reset_index(drop=True)


def clean_text(text):
    """
    텍스트 정제 함수
    
    논문 3.4절 기준:
    - 특수문자, 중복 공백, 불필요한 줄바꿈 제거
    - 기술 용어 원형 유지 (Python, PyTorch, LLM, RAG 등)
    - 복지 정보, 채용 안내 문구 제거
    
    Args:
        text: 입력 텍스트
    
    Returns:
        str: 정제된 텍스트
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    # 1. 불필요한 패턴 제거
    noise_patterns = [
        r'※.*?(?=\n|$)',  # ※로 시작하는 안내
        r'★.*?(?=\n|$)',  # ★로 시작하는 안내
        r'▶.*?(?=\n|$)',  # ▶로 시작하는 안내
        r'■.*?(?=\n|$)',  # ■로 시작하는 안내
        r'\[.*?복지.*?\]',  # 복지 관련
        r'\[.*?혜택.*?\]',  # 혜택 관련
        r'근무.*?조건.*?[:：]',  # 근무 조건
        r'급여.*?[:：]',  # 급여 정보
        r'연봉.*?[:：]',  # 연봉 정보
        r'지원.*?방법.*?[:：]',  # 지원 방법
        r'접수.*?기간.*?[:：]',  # 접수 기간
        r'문의.*?[:：]',  # 문의 정보
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 2. HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. URL 제거
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 4. 이메일 제거
    text = re.sub(r'\S+@\S+', '', text)
    
    # 5. 특수문자 정리 (기술 용어 관련 문자는 보존)
    text = re.sub(r'[^\w\s가-힣.,()/-]', ' ', text)
    
    # 6. 중복 공백 및 줄바꿈 정리
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    
    # 7. 앞뒤 공백 제거
    text = text.strip()
    
    return text


def step3_create_skill_text(df):
    """
    Step 3: skill_text 필드 생성 및 텍스트 정제
    
    주요업무 + 자격요건 + 우대사항을 하나의 텍스트로 결합
    형식: [주요업무] {내용} [자격요건] {내용} [우대사항] {내용}
    
    Args:
        df: 입력 DataFrame
    
    Returns:
        DataFrame: skill_text 필드가 추가된 데이터
    """
    print("\n" + "=" * 80)
    print("STEP 3: skill_text 필드 생성 및 텍스트 정제")
    print("=" * 80)
    
    def combine_fields(row):
        """주요업무, 자격요건, 우대사항 결합"""
        fields = []
        
        if pd.notna(row['주요업무']) and str(row['주요업무']).strip():
            cleaned = clean_text(row['주요업무'])
            if cleaned:
                fields.append(f"[주요업무] {cleaned}")
        
        if pd.notna(row['자격요건']) and str(row['자격요건']).strip():
            cleaned = clean_text(row['자격요건'])
            if cleaned:
                fields.append(f"[자격요건] {cleaned}")
        
        if pd.notna(row['우대사항']) and str(row['우대사항']).strip():
            cleaned = clean_text(row['우대사항'])
            if cleaned:
                fields.append(f"[우대사항] {cleaned}")
        
        return ' '.join(fields)
    
    print("📝 텍스트 정제 및 결합 중...")
    df['skill_text'] = df.apply(combine_fields, axis=1)
    
    # 통계 출력
    text_lengths = df['skill_text'].str.len()
    
    print(f"✅ skill_text 생성 완료")
    print(f"\n📊 skill_text 길이 통계:")
    print(f"   평균: {text_lengths.mean():.0f}자")
    print(f"   중앙값: {text_lengths.median():.0f}자")
    print(f"   최소: {text_lengths.min()}자")
    print(f"   최대: {text_lengths.max()}자")
    print(f"   표준편차: {text_lengths.std():.0f}자")
    
    # 샘플 출력
    print(f"\n📝 생성된 skill_text 샘플:")
    sample = df.iloc[0]
    print(f"   직무명: {sample['직무명']}")
    print(f"   회사: {sample['회사명']}")
    print(f"   길이: {len(sample['skill_text'])}자")
    print(f"   내용: {sample['skill_text'][:200]}...")
    
    return df


def step4_filter_short_texts(df, min_length=50):
    """
    Step 4: 짧은 텍스트 필터링
    
    세분류 판단에 필요한 설명이 부족한 공고 제외
    
    Args:
        df: 입력 DataFrame
        min_length: 최소 텍스트 길이 (기본값: 50자)
    
    Returns:
        DataFrame: 필터링된 데이터
    """
    print("\n" + "=" * 80)
    print(f"STEP 4: 짧은 텍스트 필터링 (최소 {min_length}자)")
    print("=" * 80)
    
    original_count = len(df)
    print(f"필터링 전: {original_count}건")
    
    # 텍스트 길이 계산
    df['text_length'] = df['skill_text'].str.len()
    
    # 필터링
    df_filtered = df[df['text_length'] >= min_length].copy()
    
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    
    print(f"필터링 후: {filtered_count}건")
    print(f"제거된 항목: {removed_count}건")
    
    if removed_count > 0:
        # 제거된 항목 상세
        removed = df[df['text_length'] < min_length]
        print(f"\n📋 제거된 공고 상세:")
        for idx, row in removed.iterrows():
            # 공고제목이 NaN인 경우를 대비해 str() 처리 후 슬라이싱
            # pd.notnull을 사용해 값이 있는 경우만 자르고, 없으면 "제목 없음" 표시
            title_val = row['공고제목']
            safe_title = str(title_val)[:40] if pd.notnull(title_val) else "제목 없음"
            
            print(f"   [{row['직무명']}] {row['회사명']} - {safe_title}...")
            print(f"      텍스트 길이: {row['text_length']}자")
    
    print(f"\n📊 직무별 최종 분포:")
    job_counts = df_filtered['직무명'].value_counts().sort_index()
    for job, count in job_counts.items():
        print(f"   {job}: {count}건")
    print(f"   합계: {filtered_count}건")
    
    # text_length 컬럼 제거
    df_filtered = df_filtered.drop('text_length', axis=1)
    
    return df_filtered.reset_index(drop=True)

def save_preprocessed_data(df, output_path):
    """
    전처리된 데이터 저장
    
    Args:
        df: 전처리된 DataFrame
        output_path: 출력 파일 경로
    """
    print("\n" + "=" * 80)
    print("💾 전처리 결과 저장")
    print("=" * 80)
    
    # 필요한 컬럼만 선택
    output_columns = [
        '회사명', '공고제목', '직무명',
        '주요업무', '자격요건', '우대사항',
        'skill_text','직무명1', '직무명2',
        'url', '출처'
    ]
    
    df_output = df[output_columns].copy()
    
    # CSV로 저장
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 저장 완료: {output_path}")
    print(f"   최종 데이터: {len(df_output)}건")
    print(f"   컬럼 수: {len(output_columns)}개")
    
    return df_output


def print_final_summary(df):
    """
    최종 요약 통계 출력
    
    Args:
        df: 최종 전처리된 DataFrame
    """
    print("\n" + "=" * 80)
    print("📊 전처리 완료 - 최종 요약")
    print("=" * 80)
    
    # 기본 통계
    print(f"\n✅ 최종 데이터 통계:")
    print(f"   총 건수: {len(df)}건")
    print(f"   직무 수: {df['직무명'].nunique()}개")
    print(f"   기업 수: {df['회사명'].nunique()}개")
    
    # skill_text 통계
    text_lengths = df['skill_text'].str.len()
    print(f"\n📏 skill_text 길이 통계:")
    print(f"   평균: {text_lengths.mean():.0f}자")
    print(f"   중앙값: {text_lengths.median():.0f}자")
    print(f"   최소: {text_lengths.min()}자")
    print(f"   최대: {text_lengths.max()}자")
    
    # 직무별 통계
    print(f"\n📊 직무별 최종 분포:")
    job_counts = df['직무명'].value_counts().sort_index()
    total = len(df)
    print(f"   {'직무명':<30} | 건수 | 비율")
    print(f"   {'-'*30}-+------+------")
    for job, count in job_counts.items():
        pct = (count / total) * 100
        print(f"   {job:<30} | {count:4} | {pct:5.1f}%")
    print(f"   {'-'*30}-+------+------")
    print(f"   {'합계':<30} | {total:4} | 100.0%")
    
    # 데이터 품질
    print(f"\n✅ 데이터 품질:")
    print(f"   skill_text 결측: {df['skill_text'].isna().sum()}건")
    print(f"   자격요건 결측: {df['자격요건'].isna().sum()}건")
    print(f"   우대사항 결측: {df['우대사항'].isna().sum()}건")


def postid_extraction(input_path='datas/preprocessed/preprocessed_jobposting_final.csv',
                          output_path=None):

    """
    datas/preprocessed/preprocessed_jobposting_final.csv 파일에서 url 컬럼에서 postid 추출
    https://www.work24.go.kr/wk/a/b/1500/empDetailAuthView.do?wantedAuthNo=329573&...
    wantedAuthNo= 뒤에 있는 숫자만 추출해서 post_id 컬럼으로 추가
    """
    print("\n" + "=" * 80)
    print("STEP: post_id 추출")
    print("=" * 80)

    df = pd.read_csv(input_path)

    # url에서 wantedAuthNo 숫자 추출
    df['post_id'] = df['url'].astype(str).str.extract(r'wantedAuthNo=(\d+)')

    print(f"✅ post_id 추출 완료: {df['post_id'].notna().sum()}개")
    print(f"   총 레코드: {len(df)}개")

    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 저장 완료: {output_path}")

    return df


def main():
    """메인 실행 함수"""
    
    print("\n" + "=" * 80)
    print("🚀 채용공고 데이터 전처리 파이프라인 시작")
    print("=" * 80)
    
    # 파일 경로 설정
    input_file = 'datas/raw/통합_AI세분류_7개직군_jobposting280_0331_중복제거_v2.csv'
    output_file = 'datas/preprocessed/preprocessed_jobposting_final.csv'
    
    # Step 1: 데이터 로드
    df = step1_load_data(input_file)
    
    # Step 2: 중복 제거
    df = step2_remove_duplicates(df)
    
    # Step 3: skill_text 생성 및 텍스트 정제
    df = step3_create_skill_text(df)
    
    # Step 4: 짧은 텍스트 필터링
    df = step4_filter_short_texts(df, min_length=50)
    
    # 결과 저장
    df_final = save_preprocessed_data(df, output_file)
    
    # 최종 요약
    print_final_summary(df_final)
    
    print("\n" + "=" * 80)
    print("✅ 전처리 완료!")
    print("=" * 80)
    print(f"출력 파일: {output_file}")
    print(f"최종 데이터: {len(df_final)}건")

    # postid 생성
    postid_extraction(input_path='datas/preprocessed/preprocessed_jobposting_final.csv',
                      output_path='datas/preprocessed/preprocessed_jobposting_with_postid.csv')

    
    return df_final


if __name__ == "__main__":
    df_final = main()