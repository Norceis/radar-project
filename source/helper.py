from source.preprocessHighFPS import _chunk_to_fft
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


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
    y = np.round(y, 2)
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


def get_argmaxed_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    This function acts like a treshold, with maximum value in column as a treshold value
    :param spectrogram: spectrogram to transform
    :return: spectrogram with 1 at the row in the column with max value, else 0
    """
    argmax_spectrogram = np.zeros_like(spectrogram)
    max_indices = np.argmax(spectrogram, axis=0)
    argmax_spectrogram[max_indices, np.arange(spectrogram.shape[1])] = 1
    return argmax_spectrogram


def get_tresholded_spectogram(base_spectrogram: np.ndarray,
                              argmaxed_spectrogram: np.ndarray,
                              y_step: int = 50,
                              x_step: int = 1,
                              ones_treshold: float = 0.05) -> np.ndarray:
    """
    This function acts like a treshold, but copies values from base spectrogram (prevents missing data in spectrogram)
    :param base_spectrogram: spectrogram without any changes (not differential)
    :param argmaxed_spectrogram: output from get_argmaxed_spectrogram
    :param y_step: size of the counting window in y-axis
    :param x_step: size of the counting window in x-axis
    :param ones_treshold: percent of how many pixels in a window need to be present in order
                          to classify a window as positive
    :return: a copy of base spectrogram that contains only the part that was visible in argmaxed_spectrogram
    """
    tresholded_spectrogram = np.zeros_like(argmaxed_spectrogram)
    how_many_ones_to_follow = int(y_step * x_step * ones_treshold)

    for column in range(0 + y_step, argmaxed_spectrogram.shape[1] - y_step, y_step):
        for row in range(0 + x_step, argmaxed_spectrogram.shape[0] - x_step, x_step):

            non_zeros = np.count_nonzero(argmaxed_spectrogram[row:row + x_step, column:column + y_step] == 1)
            if non_zeros > how_many_ones_to_follow:
                tresholded_spectrogram[row:row + x_step, column:column + y_step] = base_spectrogram[row:row + x_step,
                                                                                   column:column + y_step]

    tresholded_spectrogram = np.where(tresholded_spectrogram > 0, 1, 0)

    return tresholded_spectrogram

def get_spectrogram_metrics(spectrogram: np.ndarray,
                number_of_boxes: int = 1000,
                window_size: int = 100) -> tuple:
    """
    Returns means, variations, skewness and kurtosis of overlapping windows of 1D data generated from spectrogram
    :param spectrogram: spectrogram after proper cleaning operations
    :param number_of_boxes: how many boxes to put data into (they overlap)
    :param window_size: size of the 1D window that will be used to generate metrics from
    :return: a tuple of means, variations, skewness and kurtosis of given spectrogram
    """
    box_list = []
    indices_smoothed = np.argmax(spectrogram, axis=0)

    for img_column_idx in range(window_size, spectrogram.shape[1] - window_size,
                                int(spectrogram.shape[1] / number_of_boxes)):
        box_list.append(indices_smoothed[img_column_idx - window_size: img_column_idx + window_size])

    box_means = [box.mean() for box in box_list]
    box_vars = [box.var() for box in box_list]
    box_skew = [np.nan_to_num(scipy.stats.skew(box)) for box in box_list]
    box_kurt = [np.nan_to_num(scipy.stats.kurtosis(box)) for box in box_list]

    return (box_means, box_vars, box_skew, box_kurt)

def plot_metrics(means: list,
                 vars: list,
                 skew: list,
                 kurtosis: list) -> None:
    """
    Plot metrics generated by get_spectrogram_metrics
    """

    fig, axs = plt.subplots(4, figsize=(10, 8), sharex='all')
    fig.tight_layout(pad=2)

    axs[0].plot(means)
    axs[0].set_title('Mean')
    axs[1].plot(vars)
    axs[1].set_title('Variance')
    axs[2].plot(skew)
    axs[2].set_title('Skewness')
    axs[3].plot(kurtosis)
    axs[3].set_title('Kurtosis')

    plt.show()
