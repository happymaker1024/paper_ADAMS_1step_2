"""
Ko-SBERT 기반 자동 매핑 실험
1:1 매핑 (Top-1) 및 1:3 매핑 (Top-3)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
from datetime import datetime
import os

from ncs_references import NCS_REFERENCES, get_all_ncs_jobs


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


def run_kosbert_experiment(df, output_dir='outputs/kosbert'):
    """
    Ko-SBERT 실험 실행
    
    Args:
        df: Ground Truth 데이터
        output_dir: 결과 저장 디렉토리
    """
    
    print("\n" + "=" * 80)
    print("Ko-SBERT 자동 매핑 실험 시작")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 모델 로드
    print("\n📦 Ko-SBERT 모델 로드 중...")
    model = SentenceTransformer('jhgan/ko-sbert-nli')
    print("   ✅ 모델 로드 완료")
    
    # 2. NCS 참조 문서 임베딩
    print("\n🔤 NCS 참조 문서 임베딩 중...")
    ncs_jobs = get_all_ncs_jobs()
    ncs_texts = [NCS_REFERENCES[job] for job in ncs_jobs]
    ncs_embeddings = model.encode(ncs_texts, show_progress_bar=True)
    print(f"   ✅ {len(ncs_jobs)}개 세분류 임베딩 완료")
    
    # 3. 채용공고 임베딩
    print("\n🔤 채용공고 임베딩 중...")
    job_postings = df['skill_text'].tolist()
    job_embeddings = model.encode(job_postings, show_progress_bar=True)
    print(f"   ✅ {len(job_postings)}건 임베딩 완료")
    
    # 4. 유사도 계산 및 예측
    print("\n🎯 유사도 계산 및 예측 중...")
    similarities = cosine_similarity(job_embeddings, ncs_embeddings)
    
    # Top-1 예측
    top1_indices = similarities.argmax(axis=1)
    df['kosbert_top1'] = [ncs_jobs[idx] for idx in top1_indices]
    df['kosbert_top1_score'] = similarities.max(axis=1)
    
    # Top-3 예측
    top3_indices = np.argsort(similarities, axis=1)[:, -3:][:, ::-1]
    df['kosbert_top3'] = [[ncs_jobs[idx] for idx in row] for row in top3_indices]
    
    print("   ✅ 예측 완료")
    
    # 5. 성능 평가
    print("\n📊 성능 평가 중...")
    
    y_true = df['ground_truth']
    y_pred_top1 = df['kosbert_top1']
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred_top1)
    
    # Precision, Recall, F1 (Macro)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_top1, average='macro', zero_division=0
    )
    
    # Top-3 Accuracy
    top3_correct = sum([true in pred for true, pred in zip(y_true, df['kosbert_top3'])])
    top3_accuracy = top3_correct / len(df)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_top1, labels=ncs_jobs)
    
    # 결과 딕셔너리
    results = {
        'model': 'Ko-SBERT',
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
    
    # 6. 결과 저장
    print(f"\n💾 결과 저장 중...")
    
    # 예측 결과 CSV
    output_csv = os.path.join(output_dir, 'kosbert_predictions.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"   ✅ 예측 결과: {output_csv}")
    
    # 성능 지표 JSON
    output_json = os.path.join(output_dir, 'kosbert_results.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 성능 지표: {output_json}")
    
    # Confusion Matrix CSV
    cm_df = pd.DataFrame(cm, index=ncs_jobs, columns=ncs_jobs)
    output_cm = os.path.join(output_dir, 'kosbert_confusion_matrix.csv')
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
    output_per_class = os.path.join(output_dir, 'kosbert_per_class_metrics.csv')
    per_class_df.to_csv(output_per_class, index=False, encoding='utf-8-sig')
    print(f"   ✅ 세분류별 성능: {output_per_class}")
    
    print("\n" + "=" * 80)
    print("✅ Ko-SBERT 실험 완료")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    
    # Ground Truth 파일 경로
    GT_FILE = 'datas/ground_truth_labeling_template.xlsx'
    OUTPUT_DIR = 'outputs/kosbert'
    
    # Ground Truth 로드
    df = load_ground_truth(GT_FILE)
    
    # 실험 실행
    results = run_kosbert_experiment(df, OUTPUT_DIR)
    
    print(f"\n📁 결과 저장 위치: {OUTPUT_DIR}")
