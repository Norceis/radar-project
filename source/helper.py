from source.preprocessHighFPS import _chunk_to_fft
import numpy as np
import matplotlib.pyplot as plt

def gen_spectogram(frames: np.ndarray, n: int = 512, t: int = 2924, f_slope: float = 5.711) -> (np.ndarray, np.array):
    """
    frames: np.array containing all the frames to transform
    n: number of bins
    t: sampling frequency
    f_slope: Frequency slope
    """
    scale = 1 / ((29.9792458 / f_slope) / 100)  # 29... is from speed of light
    t = 1 / t
    res = []

    for frame in frames:
        yf = _chunk_to_fft(frame)
        res.append(np.abs(yf[0:n // 2]))
    res = np.array(res)
    y = np.fft.fftfreq(n, d=t)[:n // 2]
    y = y / scale
    y /= 2
    y = np.round(y,2)
    return res.T, y


def to_dB(spectogram: np.ndarray) -> np.ndarray:
    return 20 * np.log(np.abs(spectogram))


def diff_frames(frames: np.ndarray, back: int) -> np.ndarray:
    return frames[back:] - frames[:-back]


def print_spectogram(spectogram: np.ndarray, y: np.ndarray,
                     step: int = 10, depth_limit: int = None, aspect: float = 100) -> None:
    """
    spectogram: spectogram to print
    y: list of y ticks
    step: step for printing y axis label
    depth_limit: maximum depth that we want to se in spectogram
    aspect: aspect ratio of printed spectogram, to make it more visible
    """
    if depth_limit is None:
        plt.imshow(spectogram, aspect=100)
        plt.yticks(np.arange(start=0, stop=len(spectogram), step=step), y[::step])
    else:
        y_limit = np.argmax(y > depth_limit)
        plt.imshow(spectogram[:y_limit], aspect=aspect)
        plt.yticks(np.arange(start=0, stop=y_limit, step=step), y[:y_limit:step])
    plt.ylabel('Distance[m]')
    plt.xlabel('Chirp number')
    plt.colorbar()
    plt.show()

    return
