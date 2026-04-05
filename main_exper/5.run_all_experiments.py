"""
전체 실험 통합 실행 스크립트
Ko-SBERT와 GPT-4o-mini 실험을 순차 실행
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncs_references import get_all_ncs_jobs, NCS_REFERENCES


def main():
    print("=" * 80)
    print("자동 매핑 실험 통합 실행")
    print("=" * 80)
    
    print("\n📋 NCS 인공지능 세분류 (7개):")
    for i, job in enumerate(get_all_ncs_jobs(), 1):
        print(f"   {i}. {job}")
    
    print("\n" + "=" * 80)
    print("실험 1: Ko-SBERT")
    print("=" * 80)
    
    try:
        from kosbert_experiment import load_ground_truth, run_kosbert_experiment
        
        GT_FILE = 'datas/ground_truth/ground_truth_labeling_template.xlsx'
        OUTPUT_DIR_KOSBERT = 'outputs/kosbert'
        
        df = load_ground_truth(GT_FILE)
        results_kosbert = run_kosbert_experiment(df, OUTPUT_DIR_KOSBERT)
        
        print(f"\n✅ Ko-SBERT 실험 완료")
        print(f"   결과: {OUTPUT_DIR_KOSBERT}")
        
    except Exception as e:
        print(f"\n❌ Ko-SBERT 실험 실패: {str(e)}")
    
    print("\n" + "=" * 80)
    print("실험 2: GPT-4o-mini")
    print("=" * 80)
    
    try:
        from gpt4o_experiment import load_ground_truth, run_gpt4o_experiment
        
        GT_FILE = 'datas/ground_truth/ground_truth_labeling_template.xlsx'
        OUTPUT_DIR_GPT4O = 'outputs/gpt4o'
        
        df = load_ground_truth(GT_FILE)
        results_gpt4o = run_gpt4o_experiment(df, OUTPUT_DIR_GPT4O)
        
        if results_gpt4o:
            print(f"\n✅ GPT-4o-mini 실험 완료")
            print(f"   결과: {OUTPUT_DIR_GPT4O}")
        else:
            print(f"\n⚠️  GPT-4o-mini 실험 건너뜀 (API 키 필요)")
        
    except Exception as e:
        print(f"\n❌ GPT-4o-mini 실험 실패: {str(e)}")
    
    print("\n" + "=" * 80)
    print("✅ 전체 실험 완료")
    print("=" * 80)
    print("\n📁 결과 저장 위치:")
    print(f"   Ko-SBERT: outputs/kosbert")
    print(f"   GPT-4o-mini: outputs/gpt4o")


if __name__ == "__main__":
    main()
