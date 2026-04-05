"""
6-2. Confusion Matrix 시각화
히트맵 생성 및 오분류 패턴 분석
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


def load_confusion_matrices(kosbert_dir, gpt4o_dir):
    """
    Confusion Matrix 로드
    
    Args:
        kosbert_dir: Ko-SBERT 결과 디렉토리
        gpt4o_dir: GPT-4o-mini 결과 디렉토리
    
    Returns:
        tuple: (kosbert_cm, gpt4o_cm, labels)
    """
    
    # Ko-SBERT
    kosbert_cm_path = os.path.join(kosbert_dir, 'kosbert_confusion_matrix.csv')
    kosbert_cm = pd.read_csv(kosbert_cm_path, index_col=0)
    
    # GPT-4o-mini
    gpt4o_cm_path = os.path.join(gpt4o_dir, 'gpt4o_confusion_matrix.csv')
    gpt4o_cm = pd.read_csv(gpt4o_cm_path, index_col=0)
    
    labels = kosbert_cm.index.tolist()
    
    return kosbert_cm, gpt4o_cm, labels


def create_confusion_matrix_heatmap(cm, labels, title, output_path, figsize=(10, 8)):
    """
    Confusion Matrix 히트맵 생성
    
    Args:
        cm: Confusion Matrix DataFrame
        labels: 라벨 리스트
        title: 그래프 제목
        output_path: 출력 파일 경로
        figsize: 그래프 크기
    """
    
    plt.figure(figsize=figsize)
    
    # 히트맵 생성
    sns.heatmap(
        cm, 
        annot=True,  # 숫자 표시
        fmt='d',     # 정수 형식
        cmap='Blues',
        xticklabels=[label[:8] + '...' if len(label) > 8 else label for label in labels],
        yticklabels=[label[:8] + '...' if len(label) > 8 else label for label in labels],
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 저장: {output_path}")


def analyze_misclassification_patterns(cm, labels, model_name, output_path):
    """
    오분류 패턴 분석
    
    Args:
        cm: Confusion Matrix DataFrame
        labels: 라벨 리스트
        model_name: 모델명
        output_path: 출력 파일 경로
    """
    
    print(f"\n{'=' * 80}")
    print(f"{model_name} 오분류 패턴 분석")
    print("=" * 80)
    
    # Confusion Matrix를 numpy 배열로 변환
    cm_array = cm.values
    
    # 오분류 추출 (대각선 제외)
    misclassifications = []
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm_array[i, j] > 0:
                misclassifications.append({
                    'True_Label': true_label,
                    'Predicted_Label': pred_label,
                    'Count': int(cm_array[i, j]),
                    'True_Total': int(cm_array[i, :].sum())
                })
    
    # 데이터프레임 생성
    df_misclass = pd.DataFrame(misclassifications)
    
    if len(df_misclass) > 0:
        # 오분류 비율 계산
        df_misclass['Error_Rate'] = df_misclass['Count'] / df_misclass['True_Total']
        
        # 오분류 건수 기준 정렬
        df_misclass = df_misclass.sort_values('Count', ascending=False)
        
        print(f"\n📊 주요 오분류 패턴 (상위 10개):")
        print(df_misclass.head(10).to_string(index=False))
        
        # CSV 저장
        df_misclass.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 저장: {output_path}")
        
        # 가장 혼동되는 세분류 쌍
        if len(df_misclass) > 0:
            top_confusion = df_misclass.iloc[0]
            print(f"\n⚠️  가장 빈번한 오분류:")
            print(f"   {top_confusion['True_Label']} → {top_confusion['Predicted_Label']}")
            print(f"   건수: {top_confusion['Count']} ({top_confusion['Error_Rate']*100:.1f}%)")
    else:
        print("\n✅ 오분류 없음 (완벽한 분류)")
    
    return df_misclass if len(df_misclass) > 0 else None


def calculate_classification_accuracy_per_class(cm, labels):
    """
    세분류별 정확도 계산
    
    Args:
        cm: Confusion Matrix DataFrame
        labels: 라벨 리스트
    
    Returns:
        DataFrame: 세분류별 정확도
    """
    
    cm_array = cm.values
    
    class_accuracies = []
    for i, label in enumerate(labels):
        total = cm_array[i, :].sum()
        correct = cm_array[i, i]
        accuracy = correct / total if total > 0 else 0
        
        class_accuracies.append({
            '세분류': label,
            '총샘플수': int(total),
            '정확분류': int(correct),
            '오분류': int(total - correct),
            '정확도': accuracy
        })
    
    return pd.DataFrame(class_accuracies)


def main():
    """메인 실행"""
    
    print("=" * 80)
    print("6-2. Confusion Matrix 시각화 및 오분류 패턴 분석")
    print("=" * 80)
    
    # 디렉토리 설정
    KOSBERT_DIR = 'outputs/kosbert'
    GPT4O_DIR = 'outputs/gpt4o'
    OUTPUT_DIR = 'outputs/analysis'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Confusion Matrix 로드
    print("\n📂 Confusion Matrix 로드 중...")
    kosbert_cm, gpt4o_cm, labels = load_confusion_matrices(KOSBERT_DIR, GPT4O_DIR)
    print("   ✅ 로드 완료")
    
    # Ko-SBERT 히트맵
    print("\n📊 Ko-SBERT Confusion Matrix 히트맵 생성 중...")
    kosbert_heatmap_path = os.path.join(OUTPUT_DIR, 'figure_kosbert_confusion_matrix.png')
    create_confusion_matrix_heatmap(
        kosbert_cm, 
        labels, 
        'Ko-SBERT Confusion Matrix',
        kosbert_heatmap_path
    )
    
    # GPT-4o-mini 히트맵
    print("\n📊 GPT-4o-mini Confusion Matrix 히트맵 생성 중...")
    gpt4o_heatmap_path = os.path.join(OUTPUT_DIR, 'figure_gpt4o_confusion_matrix.png')
    create_confusion_matrix_heatmap(
        gpt4o_cm, 
        labels, 
        'GPT-4o-mini Confusion Matrix',
        gpt4o_heatmap_path
    )
    
    # Ko-SBERT 오분류 패턴
    kosbert_misclass_path = os.path.join(OUTPUT_DIR, 'table_kosbert_misclassification.csv')
    df_kosbert_misclass = analyze_misclassification_patterns(
        kosbert_cm, 
        labels, 
        'Ko-SBERT',
        kosbert_misclass_path
    )
    
    # GPT-4o-mini 오분류 패턴
    gpt4o_misclass_path = os.path.join(OUTPUT_DIR, 'table_gpt4o_misclassification.csv')
    df_gpt4o_misclass = analyze_misclassification_patterns(
        gpt4o_cm, 
        labels, 
        'GPT-4o-mini',
        gpt4o_misclass_path
    )
    
    # 세분류별 정확도
    print(f"\n{'=' * 80}")
    print("세분류별 분류 정확도")
    print("=" * 80)
    
    print("\n[Ko-SBERT]")
    df_kosbert_acc = calculate_classification_accuracy_per_class(kosbert_cm, labels)
    print(df_kosbert_acc.to_string(index=False))
    df_kosbert_acc.to_csv(
        os.path.join(OUTPUT_DIR, 'table_kosbert_class_accuracy.csv'),
        index=False, 
        encoding='utf-8-sig'
    )
    
    print("\n[GPT-4o-mini]")
    df_gpt4o_acc = calculate_classification_accuracy_per_class(gpt4o_cm, labels)
    print(df_gpt4o_acc.to_string(index=False))
    df_gpt4o_acc.to_csv(
        os.path.join(OUTPUT_DIR, 'table_gpt4o_class_accuracy.csv'),
        index=False, 
        encoding='utf-8-sig'
    )
    
    print("\n" + "=" * 80)
    print("✅ Confusion Matrix 시각화 완료")
    print("=" * 80)
    print(f"\n📁 결과 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()