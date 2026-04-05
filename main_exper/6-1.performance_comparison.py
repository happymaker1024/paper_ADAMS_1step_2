"""
6-1. 성능 비교표 생성
Ko-SBERT vs GPT-4o-mini 성능 비교
"""

import pandas as pd
import json
import os


def load_experiment_results(kosbert_dir, gpt4o_dir):
    """
    실험 결과 로드
    
    Args:
        kosbert_dir: Ko-SBERT 결과 디렉토리
        gpt4o_dir: GPT-4o-mini 결과 디렉토리
    
    Returns:
        tuple: (kosbert_results, gpt4o_results)
    """
    
    # Ko-SBERT 결과
    kosbert_json = os.path.join(kosbert_dir, 'kosbert_results.json')
    with open(kosbert_json, 'r', encoding='utf-8') as f:
        kosbert_results = json.load(f)
    
    # GPT-4o-mini 결과
    gpt4o_json = os.path.join(gpt4o_dir, 'gpt4o_results.json')
    with open(gpt4o_json, 'r', encoding='utf-8') as f:
        gpt4o_results = json.load(f)
    
    return kosbert_results, gpt4o_results


def create_performance_comparison_table(kosbert_results, gpt4o_results, output_path):
    """
    성능 비교표 생성 (표 5)
    
    Args:
        kosbert_results: Ko-SBERT 결과
        gpt4o_results: GPT-4o-mini 결과
        output_path: 출력 파일 경로
    """
    
    print("\n" + "=" * 80)
    print("표 5. 전체 성능 비교 결과")
    print("=" * 80)
    
    # 데이터 구성
    comparison_data = {
        '모델': ['Ko-SBERT (1:1)', 'GPT-4o-mini (1:1)', 'Ko-SBERT (1:3)', 'GPT-4o-mini (1:3)'],
        'Accuracy': [
            kosbert_results['metrics']['accuracy_top1'],
            gpt4o_results['metrics']['accuracy_top1'],
            kosbert_results['metrics']['accuracy_top3'],
            gpt4o_results['metrics']['accuracy_top3']
        ],
        'Macro Precision': [
            kosbert_results['metrics']['precision_macro'],
            gpt4o_results['metrics']['precision_macro'],
            '-',
            '-'
        ],
        'Macro Recall': [
            kosbert_results['metrics']['recall_macro'],
            gpt4o_results['metrics']['recall_macro'],
            '-',
            '-'
        ],
        'Macro F1': [
            kosbert_results['metrics']['f1_macro'],
            gpt4o_results['metrics']['f1_macro'],
            '-',
            '-'
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # 출력
    print("\n")
    print(df.to_string(index=False))
    
    # CSV 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    # 최고 성능 표시
    print("\n📊 최고 성능:")
    print(f"   Top-1 Accuracy: {df.loc[df['Accuracy'].astype(float).idxmax(), '모델']} ({df['Accuracy'].astype(float).max():.4f})")
    
    return df


def create_detailed_metrics_table(kosbert_results, gpt4o_results, output_path):
    """
    상세 성능 지표표 생성
    
    Args:
        kosbert_results: Ko-SBERT 결과
        gpt4o_results: GPT-4o-mini 결과
        output_path: 출력 파일 경로
    """
    
    print("\n" + "=" * 80)
    print("상세 성능 지표")
    print("=" * 80)
    
    detailed_data = {
        '지표': [
            'Top-1 Accuracy',
            'Top-3 Accuracy',
            'Macro Precision',
            'Macro Recall',
            'Macro F1-score'
        ],
        'Ko-SBERT': [
            f"{kosbert_results['metrics']['accuracy_top1']:.4f}",
            f"{kosbert_results['metrics']['accuracy_top3']:.4f}",
            f"{kosbert_results['metrics']['precision_macro']:.4f}",
            f"{kosbert_results['metrics']['recall_macro']:.4f}",
            f"{kosbert_results['metrics']['f1_macro']:.4f}"
        ],
        'GPT-4o-mini': [
            f"{gpt4o_results['metrics']['accuracy_top1']:.4f}",
            f"{gpt4o_results['metrics']['accuracy_top3']:.4f}",
            f"{gpt4o_results['metrics']['precision_macro']:.4f}",
            f"{gpt4o_results['metrics']['recall_macro']:.4f}",
            f"{gpt4o_results['metrics']['f1_macro']:.4f}"
        ]
    }
    
    df = pd.DataFrame(detailed_data)
    
    print("\n")
    print(df.to_string(index=False))
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    return df


def main():
    """메인 실행"""
    
    print("=" * 80)
    print("6-1. 성능 비교표 생성")
    print("=" * 80)
    
    # 디렉토리 설정
    KOSBERT_DIR = 'outputs/kosbert'
    GPT4O_DIR = 'outputs/gpt4o'
    OUTPUT_DIR = 'outputs/analysis'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 결과 로드
    print("\n📂 실험 결과 로드 중...")
    kosbert_results, gpt4o_results = load_experiment_results(KOSBERT_DIR, GPT4O_DIR)
    print("   ✅ 로드 완료")
    
    # 성능 비교표 생성
    table1_path = os.path.join(OUTPUT_DIR, 'table_performance_comparison.csv')
    df_comparison = create_performance_comparison_table(kosbert_results, gpt4o_results, table1_path)
    
    # 상세 지표표 생성
    table2_path = os.path.join(OUTPUT_DIR, 'table_detailed_metrics.csv')
    df_detailed = create_detailed_metrics_table(kosbert_results, gpt4o_results, table2_path)
    
    print("\n" + "=" * 80)
    print("✅ 성능 비교표 생성 완료")
    print("=" * 80)
    print(f"\n📁 결과 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()