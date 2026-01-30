import os
import struct
import numpy as np
import scipy.io as sio
import sys


class Extraction:
    def __init__(self):
        pass

    def __read_qstring(self, fid): # noqa
        length_raw = fid.read(4)
        if len(length_raw) < 4: return ""
        length_bytes = struct.unpack('<I', length_raw)[0]
        if length_bytes == 0xffffffff or length_bytes == 0: return ""
        data = fid.read(length_bytes)
        try:
            return data.decode('utf-16', errors='replace')
        except: # noqa
            return ""

    def convert_rhs_to_mat(self, load_path, save_folder_path):
        if os.path.isfile(load_path):
            filename = load_path
        elif os.path.isdir(load_path):
            rhs_files = [f for f in os.listdir(load_path) if f.endswith('.rhs')]
            if not rhs_files: return
            filename = os.path.join(load_path, rhs_files[-1])
        else:
            return

        if not os.path.exists(save_folder_path): os.makedirs(save_folder_path)

        with open(filename, 'rb') as fid:
            # 1. 헤더 파싱 (매트랩 구조와 완벽 일치)
            magic_number = struct.unpack('<I', fid.read(4))[0]
            if magic_number != 0xd69127ac: raise ValueError("Not a RHS file.")

            fid.read(4)  # Version
            sample_rate = struct.unpack('<f', fid.read(4))[0]
            fid.read(34)  # skip settings
            notch_mode = struct.unpack('<h', fid.read(2))[0]
            f_notch = 60.0 if notch_mode == 2 else (50.0 if notch_mode == 1 else 0)

            fid.read(24)  # skip impedance & stim step
            for _ in range(3): self.__read_qstring(fid)  # Notes
            dc_saved = struct.unpack('<h', fid.read(2))[0]
            fid.read(2)  # board_mode
            self.__read_qstring(fid)  # reference_channel

            # 2. 모든 채널 타입 카운트 (데이터 블록 크기 계산을 위해 중요)
            num_groups = struct.unpack('<h', fid.read(2))[0]
            num_amp, num_adc, num_dac, num_dig_in, num_dig_out = 0, 0, 0, 0, 0

            for _ in range(num_groups):
                self.__read_qstring(fid)
                self.__read_qstring(fid)  # group names
                group_enabled, num_ch_in_group = struct.unpack('<2h', fid.read(4))
                fid.read(2)  # signal_group_num_amp_channels

                if group_enabled:
                    for _ in range(num_ch_in_group):
                        self.__read_qstring(fid)
                        self.__read_qstring(fid)  # names
                        _, _, sig_type, ch_enabled = struct.unpack('<4h', fid.read(8))
                        fid.read(22)  # 나머지 22바이트 스킵 (8+22=30)

                        if ch_enabled:
                            if sig_type == 0:
                                num_amp += 1
                            elif sig_type == 3:
                                num_adc += 1
                            elif sig_type == 4:
                                num_dac += 1
                            elif sig_type == 5:
                                num_dig_in += 1
                            elif sig_type == 6:
                                num_dig_out += 1

            # 3. 데이터 블록 당 바이트 크기 계산
            filesize = os.path.getsize(filename)
            bytes_remaining = filesize - fid.tell()

            # 매트랩 bytes_per_block 로직
            bpb = 128 * 4  # Timestamp
            bpb += 128 * (2 + 2 + (2 if dc_saved else 0)) * num_amp  # Amp/Stim/DC
            bpb += 128 * 2 * num_adc  # ADC
            bpb += 128 * 2 * num_dac  # DAC
            if num_dig_in > 0: bpb += 128 * 2  # Dig In
            if num_dig_out > 0: bpb += 128 * 2  # Dig Out

            num_blocks = bytes_remaining // bpb
            num_samples = 128 * int(num_blocks)

            all_data = np.zeros((num_amp, num_samples), dtype=np.float32)
            timestamps = np.zeros(num_samples, dtype=np.int32)

            # 4. 데이터 로드 및 프로그레스
            print(f"데이터 로드 중 (채널: {num_amp}, 블록: {num_blocks})")
            for i in range(int(num_blocks)):
                start, end = i * 128, (i + 1) * 128
                timestamps[start:end] = np.fromfile(fid, dtype=np.int32, count=128)

                # Amp Data
                block = np.fromfile(fid, dtype=np.uint16, count=128 * num_amp)
                all_data[:, start:end] = block.reshape((128, num_amp)).T

                # DC Data (있으면 스킵)
                if dc_saved != 0: fid.read(128 * num_amp * 2)
                # Stim Data 스킵
                fid.read(128 * num_amp * 2)
                # ADC/DAC/Dig 스킵 (포인터 정렬 유지)
                fid.read(128 * (num_adc + num_dac) * 2)
                if num_dig_in > 0: fid.read(128 * 2)
                if num_dig_out > 0: fid.read(128 * 2)

                if i % 10000 == 0: sys.stdout.write(f"\rProgress: {i / num_blocks * 100:.1f}%"); sys.stdout.flush()

        # 변환 및 저장
        all_data = 0.195 * (all_data - 32768)
        t = timestamps / sample_rate
        print(f"\n최종 확인 - 시간축 범위: {t[0]:.4f}s ~ {t[-1]:.4f}s")

        for i in range(min(16, num_amp)):
            sio.savemat(os.path.join(save_folder_path, f'ch_{i + 1}.mat'),
                        {'data': {'x': t.flatten(), 'y': all_data[i, :].flatten()}})
        print("[작업 종료] 데이터 추출 완료.")