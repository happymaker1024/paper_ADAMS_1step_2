"""
6-5. 논문용 통합 표/그래프 생성
KIIT 논문 IV장에 사용할 표와 그래프 정리
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# 한글 폰트 설정 (시스템별)
def set_korean_font():
    """시스템에 맞는 한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()


def create_dataset_table(output_path):
    """
    표 3. 데이터셋 구성
    
    Args:
        output_path: 출력 파일 경로
    """
    
    print("\n" + "=" * 80)
    print("표 3. 데이터셋 구성")
    print("=" * 80)
    
    dataset_info = {
        'NCS 세분류': [
            '생성형AI엔지니어링',
            '인공지능모델링',
            '인공지능서비스기획',
            '인공지능학습데이터구축',
            '인공지능서비스구현',
            '인공지능플랫폼구축',
            '인공지능서비스운영관리',
            '합계'
        ],
        '표본 수': [49, 41, 41, 41, 37, 37, 36, 282],
        '비율(%)': [17.4, 14.5, 14.5, 14.5, 13.1, 13.1, 12.8, 100.0]
    }
    
    df = pd.DataFrame(dataset_info)
    
    print("\n")
    print(df.to_string(index=False))
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    return df


def create_experiment_setup_table(output_path):
    """
    표 4. 실험 설정
    
    Args:
        output_path: 출력 파일 경로
    """
    
    print("\n" + "=" * 80)
    print("표 4. 실험 설정")
    print("=" * 80)
    
    setup_info = {
        '항목': [
            '모델',
            '',
            'NCS 참조 문서',
            '매핑 방식',
            '',
            '평가 지표',
            '',
            '',
            '',
            'Ground Truth',
            '전체 데이터'
        ],
        '내용': [
            'Ko-SBERT (jhgan/ko-sbert-nli)',
            'GPT-4o-mini',
            'NCS 세분류명 + 직무정의 (공식 문서)',
            '1:1 매핑 (Top-1)',
            '1:3 매핑 (Top-3)',
            'Accuracy (Top-1, Top-3)',
            'Macro Precision',
            'Macro Recall',
            'Macro F1-score',
            '2인 독립 라벨링 + 합의',
            '282건 (7개 세분류)'
        ]
    }
    
    df = pd.DataFrame(setup_info)
    
    print("\n")
    print(df.to_string(index=False))
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    return df


def create_main_results_table(kosbert_results, gpt4o_results, output_path):
    """
    표 5. 전체 성능 비교 (논문용 정리)
    
    Args:
        kosbert_results: Ko-SBERT 결과
        gpt4o_results: GPT-4o-mini 결과
        output_path: 출력 파일 경로
    """
    
    print("\n" + "=" * 80)
    print("표 5. 전체 성능 비교")
    print("=" * 80)
    
    results_data = {
        '모델': ['Ko-SBERT', 'GPT-4o-mini'],
        'Accuracy@1': [
            f"{kosbert_results['metrics']['accuracy_top1']:.4f}",
            f"{gpt4o_results['metrics']['accuracy_top1']:.4f}"
        ],
        'Accuracy@3': [
            f"{kosbert_results['metrics']['accuracy_top3']:.4f}",
            f"{gpt4o_results['metrics']['accuracy_top3']:.4f}"
        ],
        'Precision': [
            f"{kosbert_results['metrics']['precision_macro']:.4f}",
            f"{gpt4o_results['metrics']['precision_macro']:.4f}"
        ],
        'Recall': [
            f"{kosbert_results['metrics']['recall_macro']:.4f}",
            f"{gpt4o_results['metrics']['recall_macro']:.4f}"
        ],
        'F1-score': [
            f"{kosbert_results['metrics']['f1_macro']:.4f}",
            f"{gpt4o_results['metrics']['f1_macro']:.4f}"
        ]
    }
    
    df = pd.DataFrame(results_data)
    
    print("\n")
    print(df.to_string(index=False))
    
    # 최고 성능 표시
    kosbert_acc = kosbert_results['metrics']['accuracy_top1']
    gpt4o_acc = gpt4o_results['metrics']['accuracy_top1']
    
    print(f"\n📊 주요 결과:")
    print(f"   최고 Accuracy@1: {'Ko-SBERT' if kosbert_acc > gpt4o_acc else 'GPT-4o-mini'} ({max(kosbert_acc, gpt4o_acc):.4f})")
    print(f"   성능 차이: {abs(kosbert_acc - gpt4o_acc):.4f} ({abs(kosbert_acc - gpt4o_acc)*100:.2f}%p)")
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    return df


def create_per_class_results_table(kosbert_per_class, gpt4o_per_class, output_path):
    """
    표 6. 세분류별 성능 (논문용 정리)
    
    Args:
        kosbert_per_class: Ko-SBERT 세분류별 성능
        gpt4o_per_class: GPT-4o-mini 세분류별 성능
        output_path: 출력 파일 경로
    """
    
    print("\n" + "=" * 80)
    print("표 6. 세분류별 F1-score")
    print("=" * 80)
    
    # 병합
    merged = pd.merge(
        kosbert_per_class[['세분류', 'F1']],
        gpt4o_per_class[['세분류', 'F1']],
        on='세분류',
        suffixes=('_KoSBERT', '_GPT4o')
    )
    
    merged.columns = ['세분류', 'Ko-SBERT F1', 'GPT-4o F1']
    
    # 정렬 (Ko-SBERT F1 기준 내림차순)
    merged = merged.sort_values('Ko-SBERT F1', ascending=False)
    
    print("\n")
    print(merged.to_string(index=False))
    
    # 성능 분석
    print(f"\n📊 세분류별 성능 분석:")
    
    kosbert_best = merged.iloc[0]
    kosbert_worst = merged.iloc[-1]
    
    print(f"\n[Ko-SBERT]")
    print(f"  최고: {kosbert_best['세분류']} (F1={kosbert_best['Ko-SBERT F1']:.4f})")
    print(f"  최저: {kosbert_worst['세분류']} (F1={kosbert_worst['Ko-SBERT F1']:.4f})")
    
    gpt4o_best_idx = merged['GPT-4o F1'].idxmax()
    gpt4o_worst_idx = merged['GPT-4o F1'].idxmin()
    
    print(f"\n[GPT-4o-mini]")
    print(f"  최고: {merged.loc[gpt4o_best_idx, '세분류']} (F1={merged.loc[gpt4o_best_idx, 'GPT-4o F1']:.4f})")
    print(f"  최저: {merged.loc[gpt4o_worst_idx, '세분류']} (F1={merged.loc[gpt4o_worst_idx, 'GPT-4o F1']:.4f})")
    
    merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    return merged


def create_combined_performance_chart(kosbert_results, gpt4o_results, output_path):
    """
    그림 1. 종합 성능 비교 차트
    
    Args:
        kosbert_results: Ko-SBERT 결과
        gpt4o_results: GPT-4o-mini 결과
        output_path: 출력 파일 경로
    """
    
    print("\n📊 그림 1. 종합 성능 비교 차트 생성 중...")
    
    metrics = ['Accuracy@1', 'Precision', 'Recall', 'F1-score']
    kosbert_values = [
        kosbert_results['metrics']['accuracy_top1'],
        kosbert_results['metrics']['precision_macro'],
        kosbert_results['metrics']['recall_macro'],
        kosbert_results['metrics']['f1_macro']
    ]
    gpt4o_values = [
        gpt4o_results['metrics']['accuracy_top1'],
        gpt4o_results['metrics']['precision_macro'],
        gpt4o_results['metrics']['recall_macro'],
        gpt4o_results['metrics']['f1_macro']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, kosbert_values, width, label='Ko-SBERT', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, gpt4o_values, width, label='GPT-4o-mini', alpha=0.8, color='coral')
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Performance Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {output_path}")


def create_accuracy_comparison_chart(kosbert_results, gpt4o_results, output_path):
    """
    그림 2. Top-1 vs Top-3 Accuracy 비교
    
    Args:
        kosbert_results: Ko-SBERT 결과
        gpt4o_results: GPT-4o-mini 결과
        output_path: 출력 파일 경로
    """
    
    print("\n📊 그림 2. Top-1 vs Top-3 Accuracy 비교 차트 생성 중...")
    
    models = ['Ko-SBERT', 'GPT-4o-mini']
    top1_values = [
        kosbert_results['metrics']['accuracy_top1'],
        gpt4o_results['metrics']['accuracy_top1']
    ]
    top3_values = [
        kosbert_results['metrics']['accuracy_top3'],
        gpt4o_results['metrics']['accuracy_top3']
    ]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars1 = ax.bar(x - width/2, top1_values, width, label='Top-1 (1:1 Mapping)', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, top3_values, width, label='Top-3 (1:3 Mapping)', alpha=0.8, color='#A23B72')
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Top-1 vs Top-3 Accuracy Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {output_path}")


def generate_paper_summary(kosbert_results, gpt4o_results, output_path):
    """
    논문 IV장 결과 요약 텍스트 생성
    
    Args:
        kosbert_results: Ko-SBERT 결과
        gpt4o_results: GPT-4o-mini 결과
        output_path: 출력 파일 경로
    """
    
    print("\n📝 논문 결과 요약 텍스트 생성 중...")
    
    kosbert_acc1 = kosbert_results['metrics']['accuracy_top1']
    kosbert_acc3 = kosbert_results['metrics']['accuracy_top3']
    kosbert_f1 = kosbert_results['metrics']['f1_macro']
    
    gpt4o_acc1 = gpt4o_results['metrics']['accuracy_top1']
    gpt4o_acc3 = gpt4o_results['metrics']['accuracy_top3']
    gpt4o_f1 = gpt4o_results['metrics']['f1_macro']
    
    summary = f"""
================================================================================
논문 IV장 실험 결과 요약
================================================================================

1. 전체 성능 비교

1:1 매핑(Top-1) 결과, Ko-SBERT는 Accuracy {kosbert_acc1:.4f}, Macro F1-score 
{kosbert_f1:.4f}을 기록하였으며, GPT-4o-mini는 Accuracy {gpt4o_acc1:.4f}, Macro 
F1-score {gpt4o_f1:.4f}을 달성하였다. 

1:3 매핑(Top-3) 결과에서는 Ko-SBERT가 {kosbert_acc3:.4f}, GPT-4o-mini가 
{gpt4o_acc3:.4f}의 정확도를 보여, Top-3 예측 시 두 모델 모두 {((kosbert_acc3 + gpt4o_acc3)/2 - (kosbert_acc1 + gpt4o_acc1)/2):.4f}p 
향상된 성능을 나타냈다.

2. 모델 간 비교

{"Ko-SBERT" if kosbert_acc1 > gpt4o_acc1 else "GPT-4o-mini"}가 Top-1 Accuracy에서 
{abs(kosbert_acc1 - gpt4o_acc1):.4f}p 더 높은 성능을 보였다. 이는 
{"임베딩 기반 유사도 측정 방식이" if kosbert_acc1 > gpt4o_acc1 else "언어모델의 맥락 이해 능력이"} 
NCS 세분류 자동 매핑 작업에 더 적합함을 시사한다.

3. 세분류별 성능 차이

일부 세분류는 두 모델 모두 높은 성능을 보인 반면, 직무 경계가 모호한 
세분류에서는 상대적으로 낮은 성능을 기록하였다. 이는 NCS 세분류 간 
업무 영역 중복이 자동 매핑의 주요 과제임을 보여준다.

4. 실용적 시사점

본 연구 결과는 생성형 AI 채용공고의 NCS 세분류 자동 매핑이 실용 가능한 
수준임을 입증하였으며, K-Digital Training 교육과정 설계 자동화의 
기초 자료로 활용될 수 있다.

================================================================================
주요 수치
================================================================================

Ko-SBERT:
  - Accuracy@1: {kosbert_acc1:.4f}
  - Accuracy@3: {kosbert_acc3:.4f}
  - Macro F1: {kosbert_f1:.4f}

GPT-4o-mini:
  - Accuracy@1: {gpt4o_acc1:.4f}
  - Accuracy@3: {gpt4o_acc3:.4f}
  - Macro F1: {gpt4o_f1:.4f}

================================================================================
"""
    
    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print(f"✅ 저장: {output_path}")


def main():
    """메인 실행"""
    
    print("=" * 80)
    print("6-5. 논문용 통합 표/그래프 생성")
    print("=" * 80)
    
    # 디렉토리 설정
    KOSBERT_DIR = 'outputs/kosbert'
    GPT4O_DIR = 'outputs/gpt4o'
    OUTPUT_DIR = 'outputs/paper'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 결과 로드
    print("\n📂 실험 결과 로드 중...")
    
    with open(os.path.join(KOSBERT_DIR, 'kosbert_results.json'), 'r', encoding='utf-8') as f:
        kosbert_results = json.load(f)
    
    with open(os.path.join(GPT4O_DIR, 'gpt4o_results.json'), 'r', encoding='utf-8') as f:
        gpt4o_results = json.load(f)
    
    kosbert_per_class = pd.read_csv(os.path.join(KOSBERT_DIR, 'kosbert_per_class_metrics.csv'))
    gpt4o_per_class = pd.read_csv(os.path.join(GPT4O_DIR, 'gpt4o_per_class_metrics.csv'))
    
    print("   ✅ 로드 완료")
    
    # 표 3: 데이터셋 구성
    table3_path = os.path.join(OUTPUT_DIR, 'table3_dataset.csv')
    create_dataset_table(table3_path)
    
    # 표 4: 실험 설정
    table4_path = os.path.join(OUTPUT_DIR, 'table4_experiment_setup.csv')
    create_experiment_setup_table(table4_path)
    
    # 표 5: 전체 성능 비교
    table5_path = os.path.join(OUTPUT_DIR, 'table5_main_results.csv')
    create_main_results_table(kosbert_results, gpt4o_results, table5_path)
    
    # 표 6: 세분류별 성능
    table6_path = os.path.join(OUTPUT_DIR, 'table6_per_class_results.csv')
    create_per_class_results_table(kosbert_per_class, gpt4o_per_class, table6_path)
    
    # 그림 1: 종합 성능 비교
    fig1_path = os.path.join(OUTPUT_DIR, 'figure1_overall_comparison.png')
    create_combined_performance_chart(kosbert_results, gpt4o_results, fig1_path)
    
    # 그림 2: Top-1 vs Top-3
    fig2_path = os.path.join(OUTPUT_DIR, 'figure2_accuracy_comparison.png')
    create_accuracy_comparison_chart(kosbert_results, gpt4o_results, fig2_path)
    
    # 결과 요약 텍스트
    summary_path = os.path.join(OUTPUT_DIR, 'paper_results_summary.txt')
    generate_paper_summary(kosbert_results, gpt4o_results, summary_path)
    
    print("\n" + "=" * 80)
    print("✅ 논문용 표/그래프 생성 완료")
    print("=" * 80)
    print(f"\n📁 결과 위치: {OUTPUT_DIR}")
    print("\n생성된 파일:")
    print("  - table3_dataset.csv (표 3. 데이터셋 구성)")
    print("  - table4_experiment_setup.csv (표 4. 실험 설정)")
    print("  - table5_main_results.csv (표 5. 전체 성능 비교)")
    print("  - table6_per_class_results.csv (표 6. 세분류별 성능)")
    print("  - figure1_overall_comparison.png (그림 1. 종합 성능 비교)")
    print("  - figure2_accuracy_comparison.png (그림 2. Top-1 vs Top-3)")
    print("  - paper_results_summary.txt (논문 결과 요약)")


if __name__ == "__main__":
    main()