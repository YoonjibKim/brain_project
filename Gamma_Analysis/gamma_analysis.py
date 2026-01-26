import os
import glob
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# GUI 없는 환경 대응
matplotlib.use('Agg')


class GammaAnalyzer:
    def __init__(self):
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'

    def _worker_process(self, task_info):
        """채널별 데이터 로드 및 고해상도 주파수 분석"""
        load_dir_path, ch_num, save_dir_path, params = task_info
        t_start, t_end = params['t_start'], params['t_end']
        bands = params['bands']

        file_path = os.path.join(load_dir_path, f'ch_{ch_num}.mat')
        # [요청] 파일이 없는 채널은 분석에서 제외
        if not os.path.exists(file_path):
            return None

        # 데이터 로드
        mat_data = sio.loadmat(file_path)
        raw_x = mat_data['data'][0, 0]['x'].flatten()
        raw_y = mat_data['data'][0, 0]['y'].flatten()
        fs = 1.0 / np.mean(np.diff(raw_x)) # noqa

        # 분석 시간 구간 필터링
        raw_x_zero = raw_x - raw_x[0]
        y_total = raw_y[(raw_x_zero >= t_start) & (raw_x_zero <= t_end)]

        # [요청] 고해상도 주파수 분석 (2초 윈도우 = 0.5Hz 정밀도)
        nperseg = int(fs * 2.0)
        noverlap = int(nperseg * 0.9)
        f, t_spec, sxx = signal.spectrogram(y_total, fs, nperseg=nperseg, noverlap=noverlap)
        t_spec_shifted = t_spec + t_start

        # [요청] 주파수 대역별 평균 Power 산출
        ch_res = {'Channel': f'Ch.{ch_num}'}
        for name, (f_min, f_max) in bands.items():
            f_mask = (f >= f_min) & (f <= f_max)
            # 전체 구간에 대한 해당 대역의 평균 에너지(Power) 계산
            ch_res[f'{name}_Avg_Power'] = np.mean(sxx[f_mask, :])

        # 스펙트로그램 시각화 (임의의 점선/수직선 코드 완전 삭제)
        self.__save_spectrogram(f, t_spec_shifted, sxx, ch_num, save_dir_path)
        return ch_res

    def analyze_gamma_bands(self, load_dir_path, save_dir_path='Gamma_Analysis/Result', t_start=0, t_end=100):
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path, exist_ok=True)

        file_list = glob.glob(os.path.join(load_dir_path, "ch_*.mat"))
        ch_list = sorted([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in file_list])

        # [요청] 대역 설정: Others(1-30), Gamma 3분할(30-300)
        params = {
            't_start': t_start, 't_end': t_end,
            'bands': {
                'Others(1-30Hz)': (1, 30),
                'Low_Gamma(30-60Hz)': (30, 60),
                'Mid_Gamma(60-90Hz)': (60, 90),
                'High_Gamma(90-300Hz)': (90, 300)
            }
        }

        results = []
        with Pool(processes=cpu_count()) as pool:
            tasks = [(load_dir_path, ch, save_dir_path, params) for ch in ch_list]
            for res in tqdm(pool.imap_unordered(self._worker_process, tasks), total=len(tasks), desc="Processing"):
                if res: results.append(res)

        # 평균 Power 보고서 저장
        df = pd.DataFrame(results).sort_values('Channel')
        df.to_csv(os.path.join(save_dir_path, 'Gamma_Power_Report.csv'), index=False)
        print(f"\n[완료] 분석 결과 저장 경로: {save_dir_path}")

    def __save_spectrogram(self, f, t, sxx, ch_num, save_dir): # noqa
        fig, ax = plt.subplots(figsize=(12, 6))
        # [요청] 1~300Hz 주파수 범위 고정
        f_mask = (f >= 1) & (f <= 300)
        pcm = ax.pcolormesh(t, f[f_mask], 10 * np.log10(sxx[f_mask, :] + 1e-12), cmap='jet', shading='auto')

        # 30초, 50초 등 임의의 수직선을 그리는 코드는 모두 삭제함
        ax.set_ylim(1, 300)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Channel {ch_num} High-Res Spectrogram')
        fig.colorbar(pcm, label='Power (dB)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'ch{ch_num}_spectrogram.png'))
        plt.close(fig)