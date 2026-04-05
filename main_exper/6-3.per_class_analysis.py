"""
6-3. 세분류별 성능 분석
F1-score 비교 및 성능 패턴 분석
"""

import pandas as pd
import numpy as np
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


def load_per_class_metrics(kosbert_dir, gpt4o_dir):
    """
    세분류별 성능 지표 로드
    
    Args:
        kosbert_dir: Ko-SBERT 결과 디렉토리
        gpt4o_dir: GPT-4o-mini 결과 디렉토리
    
    Returns:
        tuple: (kosbert_df, gpt4o_df)
    """
    
    kosbert_path = os.path.join(kosbert_dir, 'kosbert_per_class_metrics.csv')
    gpt4o_path = os.path.join(gpt4o_dir, 'gpt4o_per_class_metrics.csv')
    
    kosbert_df = pd.read_csv(kosbert_path)
    gpt4o_df = pd.read_csv(gpt4o_path)
    
    return kosbert_df, gpt4o_df


def create_per_class_comparison_table(kosbert_df, gpt4o_df, output_path):
    """
    세분류별 성능 비교표 생성 (표 6)
    
    Args:
        kosbert_df: Ko-SBERT 세분류별 성능
        gpt4o_df: GPT-4o-mini 세분류별 성능
        output_path: 출력 파일 경로
    """
    
    print("\n" + "=" * 80)
    print("표 6. 세분류별 성능 비교")
    print("=" * 80)
    
    # 병합
    comparison = pd.merge(
        kosbert_df[['세분류', '샘플수', 'Precision', 'Recall', 'F1']],
        gpt4o_df[['세분류', 'Precision', 'Recall', 'F1']],
        on='세분류',
        suffixes=('_KoSBERT', '_GPT4o')
    )
    
    # 컬럼 재정렬
    comparison = comparison[[
        '세분류', '샘플수',
        'Precision_KoSBERT', 'Recall_KoSBERT', 'F1_KoSBERT',
        'Precision_GPT4o', 'Recall_GPT4o', 'F1_GPT4o'
    ]]
    
    # 컬럼명 변경
    comparison.columns = [
        '세분류', '샘플수',
        'Ko-SBERT Precision', 'Ko-SBERT Recall', 'Ko-SBERT F1',
        'GPT-4o Precision', 'GPT-4o Recall', 'GPT-4o F1'
    ]
    
    print("\n")
    print(comparison.to_string(index=False))
    
    # 저장
    comparison.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    return comparison


def analyze_performance_patterns(comparison_df, output_path):
    """
    성능 패턴 분석
    
    Args:
        comparison_df: 세분류별 비교 데이터
        output_path: 출력 파일 경로
    """
    
    print("\n" + "=" * 80)
    print("성능 패턴 분석")
    print("=" * 80)
    
    # F1 기준 성능 분류
    kosbert_f1 = comparison_df['Ko-SBERT F1']
    gpt4o_f1 = comparison_df['GPT-4o F1']
    
    # 최고/최저 성능 세분류
    patterns = []
    
    # Ko-SBERT 최고/최저
    kosbert_best_idx = kosbert_f1.idxmax()
    kosbert_worst_idx = kosbert_f1.idxmin()
    
    patterns.append({
        '분류': 'Ko-SBERT 최고 성능',
        '세분류': comparison_df.loc[kosbert_best_idx, '세분류'],
        'F1': kosbert_f1.iloc[kosbert_best_idx],
        '샘플수': comparison_df.loc[kosbert_best_idx, '샘플수']
    })
    
    patterns.append({
        '분류': 'Ko-SBERT 최저 성능',
        '세분류': comparison_df.loc[kosbert_worst_idx, '세분류'],
        'F1': kosbert_f1.iloc[kosbert_worst_idx],
        '샘플수': comparison_df.loc[kosbert_worst_idx, '샘플수']
    })
    
    # GPT-4o 최고/최저
    gpt4o_best_idx = gpt4o_f1.idxmax()
    gpt4o_worst_idx = gpt4o_f1.idxmin()
    
    patterns.append({
        '분류': 'GPT-4o 최고 성능',
        '세분류': comparison_df.loc[gpt4o_best_idx, '세분류'],
        'F1': gpt4o_f1.iloc[gpt4o_best_idx],
        '샘플수': comparison_df.loc[gpt4o_best_idx, '샘플수']
    })
    
    patterns.append({
        '분류': 'GPT-4o 최저 성능',
        '세분류': comparison_df.loc[gpt4o_worst_idx, '세분류'],
        'F1': gpt4o_f1.iloc[gpt4o_worst_idx],
        '샘플수': comparison_df.loc[gpt4o_worst_idx, '샘플수']
    })
    
    df_patterns = pd.DataFrame(patterns)
    
    print("\n📊 성능 패턴:")
    print(df_patterns.to_string(index=False))
    
    # 저장
    df_patterns.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {output_path}")
    
    # 성능 구간별 분류
    print("\n📊 성능 구간별 세분류 분포:")
    
    for model, f1_series in [('Ko-SBERT', kosbert_f1), ('GPT-4o', gpt4o_f1)]:
        print(f"\n[{model}]")
        high = (f1_series >= 0.7).sum()
        medium = ((f1_series >= 0.5) & (f1_series < 0.7)).sum()
        low = (f1_series < 0.5).sum()
        
        print(f"  높은 성능 (F1 ≥ 0.7): {high}개")
        print(f"  중간 성능 (0.5 ≤ F1 < 0.7): {medium}개")
        print(f"  낮은 성능 (F1 < 0.5): {low}개")
    
    return df_patterns


def create_f1_comparison_chart(comparison_df, output_path):
    """
    F1-score 비교 차트 생성
    
    Args:
        comparison_df: 세분류별 비교 데이터
        output_path: 출력 파일 경로
    """
    
    print("\n📊 F1-score 비교 차트 생성 중...")
    
    # 세분류명 축약 (8자)
    labels = [label[:8] + '...' if len(label) > 8 else label 
              for label in comparison_df['세분류']]
    
    kosbert_f1 = comparison_df['Ko-SBERT F1'].values
    gpt4o_f1 = comparison_df['GPT-4o F1'].values
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, kosbert_f1, width, label='Ko-SBERT', alpha=0.8)
    bars2 = ax.bar(x + width/2, gpt4o_f1, width, label='GPT-4o-mini', alpha=0.8)
    
    ax.set_xlabel('NCS Job Categories', fontsize=12)
    ax.set_ylabel('F1-score', fontsize=12)
    ax.set_title('F1-score Comparison by Job Category', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {output_path}")


def create_performance_heatmap(comparison_df, output_path):
    """
    성능 히트맵 생성 (Precision, Recall, F1)
    
    Args:
        comparison_df: 세분류별 비교 데이터
        output_path: 출력 파일 경로
    """
    
    print("\n📊 성능 히트맵 생성 중...")
    
    # 데이터 준비
    jobs = [label[:8] + '...' if len(label) > 8 else label 
            for label in comparison_df['세분류']]
    
    # Ko-SBERT 데이터
    kosbert_data = comparison_df[['Ko-SBERT Precision', 'Ko-SBERT Recall', 'Ko-SBERT F1']].values
    
    # GPT-4o 데이터
    gpt4o_data = comparison_df[['GPT-4o Precision', 'GPT-4o Recall', 'GPT-4o F1']].values
    
    # 두 개의 서브플롯
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ko-SBERT 히트맵
    sns.heatmap(
        kosbert_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=['Precision', 'Recall', 'F1'],
        yticklabels=jobs,
        vmin=0,
        vmax=1,
        ax=ax1,
        cbar_kws={'label': 'Score'}
    )
    ax1.set_title('Ko-SBERT Performance', fontsize=12)
    
    # GPT-4o 히트맵
    sns.heatmap(
        gpt4o_data,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        xticklabels=['Precision', 'Recall', 'F1'],
        yticklabels=jobs,
        vmin=0,
        vmax=1,
        ax=ax2,
        cbar_kws={'label': 'Score'}
    )
    ax2.set_title('GPT-4o-mini Performance', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {output_path}")


def main():
    """메인 실행"""
    
    print("=" * 80)
    print("6-3. 세분류별 성능 분석")
    print("=" * 80)
    
    # 디렉토리 설정
    KOSBERT_DIR = 'outputs/kosbert'
    GPT4O_DIR = 'outputs/gpt4o'
    OUTPUT_DIR = 'outputs/analysis'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 데이터 로드
    print("\n📂 세분류별 성능 데이터 로드 중...")
    kosbert_df, gpt4o_df = load_per_class_metrics(KOSBERT_DIR, GPT4O_DIR)
    print("   ✅ 로드 완료")
    
    # 비교표 생성
    comparison_path = os.path.join(OUTPUT_DIR, 'table_per_class_comparison.csv')
    comparison_df = create_per_class_comparison_table(kosbert_df, gpt4o_df, comparison_path)
    
    # 성능 패턴 분석
    patterns_path = os.path.join(OUTPUT_DIR, 'table_performance_patterns.csv')
    df_patterns = analyze_performance_patterns(comparison_df, patterns_path)
    
    # F1-score 비교 차트
    f1_chart_path = os.path.join(OUTPUT_DIR, 'figure_f1_comparison.png')
    create_f1_comparison_chart(comparison_df, f1_chart_path)
    
    # 성능 히트맵
    heatmap_path = os.path.join(OUTPUT_DIR, 'figure_performance_heatmap.png')
    create_performance_heatmap(comparison_df, heatmap_path)
    
    print("\n" + "=" * 80)
    print("✅ 세분류별 성능 분석 완료")
    print("=" * 80)
    print(f"\n📁 결과 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()