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
from datetime import datetime

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

        # 파라미터 언패킹 (기본값 설정)
        window_sec = params.get('window_sec', 2.0)
        overlap_ratio = params.get('overlap_ratio', 0.9)

        file_path = os.path.join(load_dir_path, f'ch_{ch_num}.mat')
        if not os.path.exists(file_path):
            return None

        # 데이터 로드
        try:
            mat_data = sio.loadmat(file_path)
            raw_x = mat_data['data'][0, 0]['x'].flatten()
            raw_y = mat_data['data'][0, 0]['y'].flatten()
        except Exception as e: # noqa
            return None

        # 데이터가 너무 적을 경우 예외 처리
        if len(raw_x) < 2: return None

        fs = 1.0 / np.mean(np.diff(raw_x)) # noqa

        # 분석 시간 구간 필터링
        raw_x_zero = raw_x - raw_x[0]
        y_total = raw_y[(raw_x_zero >= t_start) & (raw_x_zero <= t_end)]

        if len(y_total) == 0: return None

        # 외부 파라미터를 적용하여 nperseg, noverlap 계산
        nperseg = int(fs * window_sec)
        noverlap = int(nperseg * overlap_ratio) # noqa

        # 데이터 길이가 윈도우보다 짧을 경우 예외 처리
        if nperseg > len(y_total):
            nperseg = len(y_total)
            noverlap = int(nperseg * overlap_ratio) # noqa
            if nperseg == 0: return None

        f, t_spec, sxx = signal.spectrogram(y_total, fs, nperseg=nperseg, noverlap=noverlap)
        t_spec_shifted = t_spec + t_start

        # 주파수 대역별 평균 Power 산출
        ch_res = {'Channel': f'Ch.{ch_num}'}

        # 시각화용 범위 계산
        all_freq_mins = [r[0] for r in bands.values()]
        all_freq_maxs = [r[1] for r in bands.values()]
        plot_f_min = min(all_freq_mins) if all_freq_mins else 1
        plot_f_max = max(all_freq_maxs) if all_freq_maxs else 300

        for name, (f_min, f_max) in bands.items():
            f_mask = (f >= f_min) & (f <= f_max)
            if np.any(f_mask):
                ch_res[f'{name}_Avg_Power'] = np.mean(sxx[f_mask, :])
            else:
                ch_res[f'{name}_Avg_Power'] = 0.0 # noqa

        # 스펙트로그램 시각화
        self.__save_spectrogram(f, t_spec_shifted, sxx, ch_num, save_dir_path, plot_f_min, plot_f_max)

        return ch_res, fs

    def analyze_gamma_bands(self, load_dir_path, save_dir_path='Gamma_Analysis/Result',
                            t_start=0, t_end=100, frequency_bands=None, window_sec=2.0, overlap_ratio=0.9):

        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path, exist_ok=True)

        file_list = glob.glob(os.path.join(load_dir_path, "ch_*.mat"))
        if not file_list:
            print(f"Warning: No .mat files found in {load_dir_path}")
            return

        ch_list = sorted([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in file_list])

        params = {
            't_start': t_start,
            't_end': t_end,
            'bands': frequency_bands,
            'window_sec': window_sec,
            'overlap_ratio': overlap_ratio
        }

        results = []
        detected_fs = None

        with Pool(processes=cpu_count()) as pool:
            tasks = [(load_dir_path, ch, save_dir_path, params) for ch in ch_list]
            for res in tqdm(pool.imap_unordered(self._worker_process, tasks), total=len(tasks), desc="Processing"):
                if res:
                    ch_data, fs_val = res
                    results.append(ch_data)
                    if detected_fs is None:
                        detected_fs = fs_val

        if results:
            # 1. 결과 데이터프레임 생성
            df = pd.DataFrame(results).sort_values('Channel')
            cols = ['Channel'] + [c for c in df.columns if c != 'Channel']
            df = df[cols]

            # 2. 메타데이터 생성 (수정된 부분: 주파수 범위 추가)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 기본 정보
            meta_dict = {
                'Experiment_Date': current_time,
                'Start_Time_sec': t_start,
                'End_Time_sec': t_end,
                'Window_sec': window_sec,
                'Overlap_ratio': overlap_ratio,
                'Fs_Hz': detected_fs if detected_fs else 'Unknown'
            }

            # 주파수 범위 정보 추가 (반복문으로 추가)
            if frequency_bands:
                for band_name, (f_min, f_max) in frequency_bands.items():
                    meta_dict[f'Band_Range_{band_name}'] = f"{f_min} ~ {f_max} Hz"

            df_metadata = pd.DataFrame(list(meta_dict.items()), columns=['Key', 'Value'])

            # 3. 파일 쓰기
            save_file = os.path.join(save_dir_path, 'Gamma_Power_Report.csv')

            with open(save_file, 'w', newline='', encoding='utf-8-sig') as f:
                f.write("=== Analysis Results ===\n")
                df.to_csv(f, index=False)

                f.write("\n\n")

                f.write("=== Experiment Metadata ===\n")
                df_metadata.to_csv(f, index=False)

            print(f"\n[완료] 분석 결과 저장 경로: {save_dir_path}")
            print(f"[정보] 메타데이터(날짜, 설정값, 주파수범위)가 파일 하단에 기록되었습니다.")
        else:
            print("\n[알림] 분석된 결과가 없습니다.")

    def __save_spectrogram(self, f, t, sxx, ch_num, save_dir, f_min, f_max): # noqa
        fig, ax = plt.subplots(figsize=(12, 6))

        f_mask = (f >= f_min) & (f <= f_max)
        if not np.any(f_mask):
            f_mask = slice(None)

        pcm = ax.pcolormesh(t, f[f_mask], 10 * np.log10(sxx[f_mask, :] + 1e-12), cmap='jet', shading='auto')

        ax.set_ylim(f_min, f_max)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Channel {ch_num} High-Res Spectrogram ({f_min}~{f_max}Hz)')
        fig.colorbar(pcm, label='Power (dB)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'ch{ch_num}_spectrogram.png'))
        plt.close(fig)