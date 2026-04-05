"""
NCS 인공지능 세분류 참조 문서 (공식 데이터 기반)
"""

import pandas as pd
import os


# NCS 공식 데이터 로드
def load_ncs_data():
    """NCS 공식 데이터 로드"""
    csv_path = 'datas/raw/ncs_info.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # 기본 데이터
        data = {
            'ncs_id': ['20010701', '20010702', '20010703', '20010704', '20010705', '20010706', '20010707'],
            'job_role_name': [
                '인공지능플랫폼구축',
                '인공지능서비스기획',
                '인공지능모델링',
                '인공지능서비스운영관리',
                '인공지능서비스구현',
                '인공지능학습데이터구축',
                '생성형AI엔지니어링'
            ],
            'job_role_desc': [
                '인공지능 플랫폼 구축은 인공지능 서비스의 요구사항을 구현하기 위해 기존 플랫폼의 활용, 인프라, 기능, 인터페이스를 구축하고 최적화하는 일이다.',
                '인공지능서비스기획은 인간의 지능으로 할 수 있는 일들을 시스템으로 구현하여 서비스로 제공하기 위한 인공지능 서비스의 목표를 설정하고 고객 요구사항 분석을 통해 인공지능 서비스 모델, 시나리오를 기획하여 실행계획을 수립하는 일이다.',
                '인공지능 모델링이란 기획된 인공지능 서비스의 목적을 달성하기 위하여, 학습 데이터를 확보, 가공, 특징 추출, 품질 검증, 학습을 통해 최적화된 모델을 도출하고 활용하는 일이다.',
                '인공지능서비스운영관리는 구축된 인공지능서비스를 체계적으로 운영하기 위하여 운영계획에 따라 품질을 유지하고 서비스를 모니터링하며 관리하는 일이다.',
                '인공지능서비스구현은 기획 목적에 부합하는 인공지능서비스를 구축하기 위해 모델링 결과를 플랫폼 환경에서 분석, 설계, 개발, 테스트, 이행하는 일이다.',
                '인공지능 학습데이터 구축은 도메인 영역별 인공지능 학습데이터 제공을 위하여 학습데이터 구축을 기획하고, 학습데이터의 획득, 저장, 라벨링, 결합, 변환, 품질 검증, 딜리버리 등을 수행하여 인공지능 학습데이터를 구축하는 일이다.',
                '생성형AI엔지니어링은 다양한 산업분야에서 비즈니스 요구에 따라 생성형 AI 모델을 선정하고 학습시켜, 맞춤형 프로덕트를 제작하고 검증하는 전 과정을 수행하는 일이다.'
            ]
        }
        return pd.DataFrame(data)


# NCS 데이터 로드
_ncs_df = load_ncs_data()

# NCS 참조 문서 딕셔너리 생성
NCS_REFERENCES = {}
for _, row in _ncs_df.iterrows():
    job_name = row['job_role_name']
    job_desc = row['job_role_desc'].strip()
    
    # 참조 텍스트 생성
    ref_text = f"""세분류명: {job_name}
직무정의: {job_desc}"""
    
    NCS_REFERENCES[job_name] = ref_text


def get_ncs_reference(job_name):
    """
    특정 세분류의 NCS 참조 문서 반환
    
    Args:
        job_name: 세분류명
    
    Returns:
        str: 참조 문서
    """
    return NCS_REFERENCES.get(job_name, "")


def get_all_ncs_jobs():
    """
    모든 NCS 세분류 목록 반환
    
    Returns:
        list: 세분류명 리스트
    """
    return list(NCS_REFERENCES.keys())


def get_ncs_dataframe():
    """
    NCS 데이터프레임 반환
    
    Returns:
        pd.DataFrame: NCS 데이터
    """
    return _ncs_df.copy()


if __name__ == "__main__":
    print("=" * 80)
    print("NCS 인공지능 세분류 참조 문서 (공식 데이터)")
    print("=" * 80)
    
    for i, (job, ref) in enumerate(NCS_REFERENCES.items(), 1):
        print(f"\n{i}. {job}")
        print("-" * 80)
        print(ref)
