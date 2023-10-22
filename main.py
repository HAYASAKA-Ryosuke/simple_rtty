import numpy as np
from typing import List
from scipy.signal import hilbert

BAUDOT_CODE = {
    'LTRS': {
        '00000': 'NUL',
        '00001': 'E',
        '00010': 'LF',
        '01000': 'CR',
        '00100': ' ',
        '00001': 'E',
        '10111': 'Q',
        '10011': 'W',
        '01010': 'R',
        '10000': 'T',
        '10101': 'Y',
        '00111': 'U',
        '00110': 'I',
        '11000': 'O',
        '10110': 'P',
        '00011': 'A',
        '00101': 'S',
        '01001': 'D',
        '01011': 'J',
        '01100': 'N',
        '01101': 'F',
        '01110': 'C',
        '01111': 'K',
        '10001': 'Z',
        '10010': 'L',
        '10100': 'H',
        '11100': 'B',
        '11101': 'X',
        '11110': 'V',
        '11011': 'FIGS',  # 数字・特殊文字モードへの切り替え
        '11111': 'LTRS',  # 文字モードへの切り替え
    },
    'FIGS': {
        '00001': '3',
        '00010': 'LF',
        '00011': '-',
        '00100': ' ',
        '00101': "'",
        '00110': '8',
        '00111': '7',
        '01000': 'CR',
        '01001': 'ENQ',
        '01010': '4',
        '01011': 'BELL',
        '01100': ',',
        '01101': '?',  # 保留域
        '01110': ':',
        '01111': '(',
        '10000': '5',
        '10001': '+',
        '10010': ')',
        '10011': '2',
        '10100': '?',  # 保留域
        '10101': '6',
        '10110': '0',
        '10111': '1',
        '11000': '9',
        '11001': '7',
        '11010': '?',  # 保留域
        '11011': 'FIGS',  # 数字・特殊文字モードへの切り替え
        '11100': '.',
        '11101': '/',
        '11110': ';',
        '11111': 'LTRS',  # 文字モードへの切り替え
    }
}

MODE = 'LTRS'


def instantaneous_frequency(signal: List, fs: int) -> List[float]:
    """
    瞬時周波数を求める
    """
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi)) * fs
    return instantaneous_frequency


def convert_bits_to_string(bits: List[bool]) -> str:
    """
    与えられた5ビット配列から文字列に変換
    """
    result = ''
    for i in range(0, len(bits), 5):
        result += BAUDOT_CODE[MODE][''.join(map(str, bits[i:i+5]))]
    return result


def decode_rtty(signal: List[float], fs: int, baudrate: float=45.45, mark_freq: int=2125, space_freq: int=2295) -> str:
    """
    RTTYをデコードする
    瞬時周波数で各ビットの周波数を求める｡マーク周波数とスペース周波数の中間の周波数からマークとスペースを判定する
    """
    # 瞬時周波数
    inst_freq = instantaneous_frequency(signal, fs)

    # マークとスペースの中間の周波数を計算
    threshold_freq = (mark_freq + space_freq) / 2.0

    # マークとスペース
    mark_space_bits = list(map(int, inst_freq <= threshold_freq))

    # 適切なサンプリング間隔で抽出
    samples_per_bit = int(fs / baudrate)
    offset = samples_per_bit // 2

    sampled_bits = mark_space_bits[offset::samples_per_bit]

    # スタートビットとストップビットの検出
    decoded_bits = []
    is_start = False
    i = 0
    for _ in range(len(sampled_bits)):
        if len(sampled_bits) <= i:
            break
        if is_start and len(sampled_bits) >= i + 6:
            if sampled_bits[i+6] and sampled_bits[i+5]:
                decoded_bits.extend(sampled_bits[i:i+5])
                i += 6
            is_start = False
        elif not sampled_bits[i] and not is_start:
            is_start = True
        i += 1
    return convert_bits_to_string(decoded_bits)


def generate_rtty_signal(message: str, fs: int, baudrate: float=45.45, mark_freq: int=2125, space_freq: int=2295) -> List[float]:
    """
    テスト用のRTTY信号生成
    """

    t_bit = 1.0 / baudrate
    samples_per_bit = int(fs * t_bit)

    signal = np.array([])
    BAUDOT_CODE_REVERSE = {v: k for k, v in BAUDOT_CODE[MODE].items()}
    for char in message:
        # スタートビットを追加
        t = np.linspace(0, t_bit, samples_per_bit, endpoint=False)
        sinewave = np.sin(2 * np.pi * space_freq * t)
        signal = np.concatenate((signal, sinewave))
        bits = [int(bit) for bit in BAUDOT_CODE_REVERSE[char]]

        for bit in bits:
            freq = mark_freq if bit else space_freq
            t = np.linspace(0, t_bit, samples_per_bit, endpoint=False)
            sinewave = np.sin(2 * np.pi * freq * t)
            signal = np.concatenate((signal, sinewave))

        # ストップビットを追加
        t = np.linspace(0, t_bit, samples_per_bit, endpoint=False)
        sinewave = np.sin(2 * np.pi * mark_freq * t)
        signal = np.concatenate((signal, sinewave))
        signal = np.concatenate((signal, sinewave))

    return signal

# RTTY信号を生成
message = "HELLO WORLD"
fs = 8000
signal = generate_rtty_signal(message, fs)

# デコードして文字列を求める
decoded_message = decode_rtty(signal, fs)
print(decoded_message)
