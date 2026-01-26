from Data.Extraction.extraction import Extraction
from Gamma_Analysis.gamma_analysis import GammaAnalyzer


def analysis(_rhs_load_path=None, _mat_save_dir=None, _gamma_save_dir=None):
    # 1. 원본 데이터 추출 (RHS -> MAT)
    if _rhs_load_path and _mat_save_dir:
        print(f"--- [1단계] 데이터 추출 시작: {_rhs_load_path} ---")
        extractor = Extraction()
        extractor.convert_rhs_to_mat(_rhs_load_path, _mat_save_dir)
    else:
        print("경고: 데이터 추출 경로 설정이 누락되었습니다.")

    # 2. 고해상도 LFP 및 감마 파워 분석
    if _mat_save_dir and _gamma_save_dir:
        print(f"--- [2단계] 고해상도 LFP/감마 분석 시작 ---")
        analyzer = GammaAnalyzer()

        # 이로써 임의의 점선이 사라지고 전체 구간의 평균 파워가 산출됩니다.
        analyzer.analyze_gamma_bands(
            load_dir_path=_mat_save_dir,
            save_dir_path=_gamma_save_dir,
            t_start=0,
            t_end=100
        )
        print(f"--- 분석 완료! 결과 확인: {_gamma_save_dir} ---")
    else:
        print("경고: 분석 파일 또는 저장 경로가 누락되었습니다.")


if __name__ == "__main__":
    rhs_path = 'Data/Extraction/Raw_Data/20241211_#5(sal)_after_merge.rhs'
    mat_path = 'Data/Extraction/Mat_Data/20241211_#5(sal)_after_merge'
    gamma_path = 'Gamma_Analysis/Result/20241211_#5(sal)_after_merge'

    # rhs_path = 'Data/Extraction/Raw_Data/20241219_#3(3mg)_after_merge.rhs'
    # mat_path = 'Data/Extraction/Mat_Data/20241219_#3(3mg)_after_merge'
    # gamma_path = 'Gamma_Analysis/Result/20241219_#3(3mg)_after_merge'

    analysis(
        _rhs_load_path=rhs_path,
        _mat_save_dir=mat_path,
        _gamma_save_dir=gamma_path
    )
