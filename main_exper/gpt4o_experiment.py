"""
GPT-4o-mini 기반 자동 매핑 실험
1:1 매핑 (Top-1) 및 1:3 매핑 (Top-3)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
import os
import time
from openai import OpenAI

from ncs_references import NCS_REFERENCES, get_all_ncs_jobs

# pip install python-dotenv
from dotenv import load_dotenv  
# 1. .env 파일의 환경 변수를 로드
load_dotenv(override=True, dotenv_path='../.env')  # 이미 설정된 환경 변수도 덮어쓰기

# 2. 환경 변수에서 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")


def load_ground_truth(file_path):
    """
    Ground Truth 데이터 로드
    
    Args:
        file_path: Excel 또는 CSV 파일 경로
    
    Returns:
        pd.DataFrame: Ground Truth 데이터
    """
    print(f"📂 Ground Truth 로드: {file_path}")
    
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, sheet_name='라벨링')
    else:
        df = pd.read_csv(file_path)
    
    # 최종합의_라벨이 있으면 사용, 없으면 직무 사용
    if '최종합의_라벨' in df.columns and df['최종합의_라벨'].notna().any():
        df['ground_truth'] = df['최종합의_라벨']
        print("   ✅ '최종합의_라벨' 사용")
    else:
        df['ground_truth'] = df['직무']
        print("   ⚠️  '최종합의_라벨' 없음 → '직무' 사용")
    
    # 결측치 제거
    df = df[df['ground_truth'].notna()].copy()
    
    print(f"   총 {len(df)}건")
    return df


def create_prompt(skill_text, ncs_jobs):
    """
    GPT-4o-mini용 프롬프트 생성
    
    Args:
        skill_text: 채용공고 텍스트
        ncs_jobs: NCS 세분류 리스트
    
    Returns:
        str: 프롬프트
    """
    
    # NCS 세분류 설명
    ncs_descriptions = ""
    for i, job in enumerate(ncs_jobs, 1):
        ref = NCS_REFERENCES[job]
        # 첫 3줄만 사용 (간략화)
        short_desc = '\n'.join(ref.strip().split('\n')[:3])
        ncs_descriptions += f"{i}. {job}\n{short_desc}\n\n"
    
    prompt = f"""다음 채용공고를 분석하여 가장 적합한 NCS 인공지능 세분류를 선택하세요.

채용공고:
{skill_text}

NCS 인공지능 세분류:
{ncs_descriptions}

지시사항:
1. 채용공고의 주요업무, 자격요건, 우대사항을 종합적으로 분석하세요.
2. 가장 적합한 세분류 1개를 선택하세요.
3. 추가로 관련성이 높은 세분류 2개를 선택하세요.
4. 반드시 JSON 형식으로만 응답하세요.

출력 형식:
{{
  "primary": "세분류명",
  "candidates": ["세분류명1", "세분류명2"]
}}

응답:"""
    
    return prompt


def classify_with_gpt4o(client, skill_text, ncs_jobs, max_retries=3):
    """
    GPT-4o-mini로 분류
    
    Args:
        client: OpenAI 클라이언트
        skill_text: 채용공고 텍스트
        ncs_jobs: NCS 세분류 리스트
        max_retries: 최대 재시도 횟수
    
    Returns:
        tuple: (primary, candidates)
    """
    
    prompt = create_prompt(skill_text, ncs_jobs)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 채용공고를 NCS 세분류로 분류하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # JSON 파싱
            # 마크다운 코드 블록 제거
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            result = json.loads(result_text)
            
            primary = result.get('primary', '')
            candidates = result.get('candidates', [])
            
            # 유효성 검증
            if primary in ncs_jobs and all(c in ncs_jobs for c in candidates):
                return primary, candidates
            else:
                print(f"   ⚠️  잘못된 응답 (시도 {attempt+1}/{max_retries})")
                
        except Exception as e:
            print(f"   ⚠️  오류 발생 (시도 {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(1)
    
    # 실패 시 첫 번째 세분류 반환
    return ncs_jobs[0], [ncs_jobs[1], ncs_jobs[2]]


def run_gpt4o_experiment(df, output_dir='outputs/gpt4o', api_key=None):
    """
    GPT-4o-mini 실험 실행
    
    Args:
        df: Ground Truth 데이터
        output_dir: 결과 저장 디렉토리
        api_key: OpenAI API 키
    """
    
    print("\n" + "=" * 80)
    print("GPT-4o-mini 자동 매핑 실험 시작")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # OpenAI 클라이언트 초기화
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("⚠️  OPENAI_API_KEY 환경변수를 설정하거나 api_key 인자를 전달하세요.")
        return None
    
    client = OpenAI(api_key=api_key)
    
    ncs_jobs = get_all_ncs_jobs()
    
    # 예측 수행
    print(f"\n🤖 GPT-4o-mini로 {len(df)}건 예측 중...")
    
    predictions_top1 = []
    predictions_top3 = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"   진행: {idx}/{len(df)}")
        
        skill_text = row['skill_text']
        primary, candidates = classify_with_gpt4o(client, skill_text, ncs_jobs)
        
        predictions_top1.append(primary)
        predictions_top3.append([primary] + candidates[:2])
        
        # API 속도 제한 고려
        time.sleep(0.5)
    
    df['gpt4o_top1'] = predictions_top1
    df['gpt4o_top3'] = predictions_top3
    
    print("   ✅ 예측 완료")
    
    # 성능 평가
    print("\n📊 성능 평가 중...")
    
    y_true = df['ground_truth']
    y_pred_top1 = df['gpt4o_top1']
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred_top1)
    
    # Precision, Recall, F1 (Macro)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_top1, average='macro', zero_division=0
    )
    
    # Top-3 Accuracy
    top3_correct = sum([true in pred for true, pred in zip(y_true, df['gpt4o_top3'])])
    top3_accuracy = top3_correct / len(df)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_top1, labels=ncs_jobs)
    
    # 결과 딕셔너리
    results = {
        'model': 'GPT-4o-mini',
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(df),
        'metrics': {
            'accuracy_top1': float(accuracy),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'accuracy_top3': float(top3_accuracy)
        },
        'confusion_matrix': cm.tolist(),
        'labels': ncs_jobs
    }
    
    print("\n✅ 평가 완료")
    print(f"   Accuracy (Top-1): {accuracy:.4f}")
    print(f"   Precision (Macro): {precision:.4f}")
    print(f"   Recall (Macro): {recall:.4f}")
    print(f"   F1 (Macro): {f1:.4f}")
    print(f"   Accuracy (Top-3): {top3_accuracy:.4f}")
    
    # 결과 저장
    print(f"\n💾 결과 저장 중...")
    
    # 예측 결과 CSV
    output_csv = os.path.join(output_dir, 'gpt4o_predictions.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"   ✅ 예측 결과: {output_csv}")
    
    # 성능 지표 JSON
    output_json = os.path.join(output_dir, 'gpt4o_results.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 성능 지표: {output_json}")
    
    # Confusion Matrix CSV
    cm_df = pd.DataFrame(cm, index=ncs_jobs, columns=ncs_jobs)
    output_cm = os.path.join(output_dir, 'gpt4o_confusion_matrix.csv')
    cm_df.to_csv(output_cm, encoding='utf-8-sig')
    print(f"   ✅ Confusion Matrix: {output_cm}")
    
    # 세분류별 성능
    per_class_metrics = []
    for i, job in enumerate(ncs_jobs):
        mask = y_true == job
        if mask.sum() > 0:
            # 이진 분류로 변환 (해당 직무 vs 나머지)
            y_true_binary = (y_true == job).astype(int)
            y_pred_binary = (y_pred_top1 == job).astype(int)
            
            job_precision, job_recall, job_f1, _ = precision_recall_fscore_support(
                y_true_binary[mask], y_pred_binary[mask], average='binary', 
                zero_division=0
            )
            per_class_metrics.append({
                '세분류': job,
                '샘플수': int(mask.sum()),
                'Precision': float(job_precision),
                'Recall': float(job_recall),
                'F1': float(job_f1)
            })
    
    per_class_df = pd.DataFrame(per_class_metrics)
    output_per_class = os.path.join(output_dir, 'gpt4o_per_class_metrics.csv')
    per_class_df.to_csv(output_per_class, index=False, encoding='utf-8-sig')
    print(f"   ✅ 세분류별 성능: {output_per_class}")
    
    print("\n" + "=" * 80)
    print("✅ GPT-4o-mini 실험 완료")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    
    # Ground Truth 파일 경로
    GT_FILE = 'datas/ground_truth/ground_truth_labeling_template.xlsx'
    OUTPUT_DIR = 'outputs/gpt4o'
    
    # Ground Truth 로드
    df = load_ground_truth(GT_FILE)
    
    # 실험 실행 (OPENAI_API_KEY 환경변수 필요)
    results = run_gpt4o_experiment(df, OUTPUT_DIR)
    
    if results:
        print(f"\n📁 결과 저장 위치: {OUTPUT_DIR}")
