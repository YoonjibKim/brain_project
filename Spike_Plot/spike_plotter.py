import os
import glob
import scipy.io as sio
import matplotlib.pyplot as plt


class SpikePlotter:
    def __init__(self):
        # GUI 없는 서버 환경 대응
        import matplotlib
        matplotlib.use('Agg')

    def plot_roi_raw(self, load_dir_path, save_dir_path, t_start=0, t_end=100): # noqa
        """
        지정된 시간 구간(ROI)의 Raw Signal(전압 파형)을 그려서 저장
        """
        # 저장 경로 생성
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        # .mat 파일 목록 로드
        mat_files = sorted(glob.glob(os.path.join(load_dir_path, "ch_*.mat")))

        if not mat_files:
            print(f"[경고] {load_dir_path} 경로에서 .mat 파일을 찾을 수 없습니다.")
            return

        print(f"--- Spike Graph(Raw Signal) 그리기 시작 ({len(mat_files)}개 채널) ---")

        for file_path in mat_files:
            try:
                # 1. 데이터 로드
                mat_data = sio.loadmat(file_path)

                # extraction.py 구조: data -> x(시간), y(전압 uV)
                raw_x = mat_data['data'][0, 0]['x'].flatten()
                raw_y = mat_data['data'][0, 0]['y'].flatten()

                # 2. ROI (관심 시간 구간) 데이터 필터링
                mask = (raw_x >= t_start) & (raw_x <= t_end)
                x_roi = raw_x[mask]
                y_roi = raw_y[mask]

                if len(x_roi) == 0:
                    print(f"스킵: 해당 시간대 데이터 없음 ({os.path.basename(file_path)})")
                    continue

                # 3. 그래프 그리기 (Time Domain)
                fig, ax = plt.subplots(figsize=(12, 6))

                # 검은색 실선으로 파형 시각화
                ax.plot(x_roi, y_roi, color='black', linewidth=0.5)

                ax.set_title(f'Raw Spike Signal: {os.path.basename(file_path)} ({t_start}s ~ {t_end}s)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (uV)')
                ax.grid(True, linestyle='--', alpha=0.5)  # 격자 표시

                # 4. 이미지 저장
                file_name = os.path.basename(file_path).replace('.mat', '_spike.png')
                save_full_path = os.path.join(save_dir_path, file_name)

                plt.tight_layout()
                plt.savefig(save_full_path)
                plt.close(fig)  # 메모리 해제

                print(f"저장 완료: {save_full_path}")

            except Exception as e:
                print(f"에러 발생 ({os.path.basename(file_path)}): {e}")