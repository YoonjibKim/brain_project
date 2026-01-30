from Data.Extraction.extraction import Extraction
from Spike_Plot.spike_plotter import SpikePlotter
from Gamma_Analysis.gamma_analysis import GammaAnalyzer


def analysis(_rhs_load_path=None, _mat_save_dir=None, _gamma_save_dir=None, _frequency_path=None, # noqa
             t_start=0, t_end=100, frequency_bands=None, window_sec=2.0, overlap_ratio=0.9): # noqa
    """
    통합 분석 함수
    :param frequency_bands: 분석할 주파수 대역 딕셔너리
    :param window_sec: STFT 윈도우 시간 (초)
    :param overlap_ratio: 오버랩 비율 (0~1)
    """

    # 1. 원본 데이터 추출 (RHS -> MAT)
    if _rhs_load_path and _mat_save_dir:
        print(f"--- 데이터 추출 시작: {_rhs_load_path} ---")
        extractor = Extraction()
        extractor.convert_rhs_to_mat(_rhs_load_path, _mat_save_dir)

    # 3. Raw Signal 스파이크 그래프 (시각적 확인용)
    if _mat_save_dir and _frequency_path:
        scanner = SpikePlotter()
        scanner.plot_roi_raw(load_dir_path=_mat_save_dir, save_dir_path=_frequency_path, t_start=t_start, t_end=t_end)
        print("--- 모든 채널 그래프 저장 완료 ---")

    # 2. 고해상도 LFP 및 감마 파워 분석
    if _mat_save_dir and _gamma_save_dir:
        print(f"--- 고해상도 LFP/감마 분석 시작 ---")
        analyzer = GammaAnalyzer()

        analyzer.analyze_gamma_bands(
            load_dir_path=_mat_save_dir,
            save_dir_path=_gamma_save_dir,
            t_start=t_start,
            t_end=t_end,
            frequency_bands=frequency_bands,
            window_sec=window_sec,
            overlap_ratio=overlap_ratio
        )
        print(f"--- 분석 완료! 결과 확인: {_gamma_save_dir} ---")


if __name__ == "__main__":
    # --- [경로 설정] ---
    # 20241211_#5(sal)_after_merge
    # rhs_path = 'Data/Extraction/Raw_Data/20241211_#5(sal)_after_merge.rhs'
    # mat_path = 'Data/Extraction/Mat_Data/20241211_#5(sal)_after_merge'
    # gamma_path = 'Gamma_Analysis/Result/20241211_#5(sal)_after_merge'
    # frequency_path = 'Spike_Plot/Result/20241211_#5(sal)_after_merge'

    # 20241219_#3(3mg)_after_merge
    rhs_path = 'Data/Extraction/Raw_Data/20241219_#3(3mg)_after_merge.rhs'
    mat_path = 'Data/Extraction/Mat_Data/20241219_#3(3mg)_after_merge'
    gamma_path = 'Gamma_Analysis/Result/20241219_#3(3mg)_after_merge'
    frequency_path = 'Spike_Plot/Result/20241219_#3(3mg)_after_merge'

    # 1. 분석할 주파수 대역 정의
    my_bands = {
        'Others': (1, 30), # 1, 30
        'Low_Gamma': (30, 60), # 30, 60
        'Mid_Gamma': (60, 90), # 60, 90
        'High_Gamma': (90, 300) # 90, 300
    }

    # 2. STFT 정밀도 설정
    win_sec = 2.0 # 2.0
    overlap = 0.9 # 0.9

    # 3. 시간 구간 설정 (초 단위)
    start_time = 100 # 100
    end_time = 200 # 200

    # --- [실행] ---
    analysis(
        _rhs_load_path=None, # rhs_path,
        _mat_save_dir=mat_path,
        _gamma_save_dir=gamma_path,
        _frequency_path=frequency_path,
        t_start=start_time,
        t_end=end_time,
        frequency_bands=my_bands,
        window_sec=win_sec,
        overlap_ratio=overlap
    )