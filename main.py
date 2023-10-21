import numpy as np
from typing import List
from scipy.signal import hilbert


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
    与えられたビット配列から文字列に変換
    """
    byte_chunks = [bits[i:i+8] for i in range(0, len(bits), 8)]
    # ビット列を文字に変換
    result = ''
    for byte_chunk in byte_chunks:
        # ビット配列を整数に変換(0b00101→5)
        byte_val = sum([bit * (2**index) for index, bit in enumerate(byte_chunk[::-1])])
        # 整数をASCII文字に変換
        result += chr(byte_val)

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

    # マークとスペースをTrue,Falseで表現
    mark_space_bits = inst_freq <= threshold_freq

    # 適切なサンプリング間隔で抽出
    samples_per_bit = int(fs / baudrate)
    offset = samples_per_bit // 2
    decoded_bits = mark_space_bits[offset::samples_per_bit]

    return convert_bits_to_string(decoded_bits)


def generate_rtty_signal(message: str, fs: int, baudrate: float=45.45, mark_freq: int=2125, space_freq: int=2295) -> List[float]:
    """
    テスト用のRTTY信号生成
    """

    t_bit = 1.0 / baudrate
    samples_per_bit = int(fs * t_bit)

    # メッセージをビット配列に変換(Aなら0b01000001なので[0, 1, 0, 0, 0, 0, 0, 1]になる)
    bits = [int(bit) for char in message for bit in format(ord(char), '08b')]

    signal = np.array([])
    for bit in bits:
        freq = mark_freq if bit else space_freq
        t = np.linspace(0, t_bit, samples_per_bit, endpoint=False)
        sinewave = np.sin(2 * np.pi * freq * t)
        signal = np.concatenate((signal, sinewave))

    return signal


if __name__ == '__main__':
    # RTTY信号を生成
    message = "Hello World!"
    fs = 8000
    signal = generate_rtty_signal(message, fs)
    
    # デコードして文字列を求める
    decoded_message = decode_rtty(signal, fs)
    print(decoded_message)

