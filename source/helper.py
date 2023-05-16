from scipy.ndimage import median_filter

from source.preprocessHighFPS import _chunk_to_fft
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from filterpy.kalman import KalmanFilter


def gen_spectogram(frames: np.ndarray, n: int = 512, t: int = 2924, f_slope: float = 5.711,
                   depth_limit: int = None) -> (np.ndarray, np.array):
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
        res.append(np.abs(yf[0:n // 2]))  # to pozostawia amplitude
    res = np.array(res)
    y = np.fft.fftfreq(n, d=t)[:n // 2]
    y = y / scale
    y /= 2
    y = np.round(y, 2)

    if depth_limit:
        y_limit = np.argmax(y > depth_limit)
        return res.T[:y_limit], y[:y_limit]
    else:
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


def gen_velocity_spectogram(frames: np.ndarray, n: int = 512, t: int = 2924, f_slope: float = 5.711) -> (
        np.ndarray, np.array):
    """
    frames: np.array containing all the frames to transform
    n: number of bins
    t: sampling frequency
    f_slope: Frequency slope
    """
    scale = 1 / ((29.9792458 / f_slope) / 100)  # 29... is from speed of light
    t = 1 / t
    y = np.fft.fftfreq(n, d=t)[:n // 2]
    y = y / scale
    y /= 2
    y = np.round(y, 2)

    first_fft = np.fft.fft(frames.imag, axis=1)
    second_fft = np.fft.fft(first_fft, axis=0)
    second_fft = np.fft.fftshift(second_fft, axes=0)

    c = 3e8  # Speed of light (m/s)

    start_freq = 77  # Starting frequency of the chirp (GHz)
    idle_time = 1000  # Time before starting next chirp (us)
    ramp_end_time = 182.52  # Time after sending each chirp (us)

    velocity_res = c / (2 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * frames.shape[1])
    # print(f'Velocity Resolution: {velocity_res} [meters/second]')

    # Apply the velocity resolution factor to the doppler indicies
    velocities = np.arange(frames.shape[1]) - (frames.shape[1] // 2)
    velocities = velocities * velocity_res

    return np.abs(second_fft.T), y, velocities


def print_vel_spectogram(spectogram: np.ndarray, y: np.ndarray, x: np.ndarray = None,
                         depth_limit: int = None, aspect: float = 100) -> None:
    """
    spectogram: spectogram to print
    y: list of y ticks
    x: list of x ticks (kinda, need to refactor this)
    step: step for printing y axis label
    depth_limit: maximum depth that we want to se in spectogram
    aspect: aspect ratio of printed spectogram, to make it more visible
    """
    if x is None:
        x = np.arange(0, spectogram.shape[1], 1)

    if depth_limit is None:
        plt.imshow(spectogram, extent=[x.min(), x.max(), y.max(), y.min()], aspect=aspect)
    else:
        y_limit = np.argmax(y > depth_limit)
        plt.imshow(spectogram[:y_limit], extent=[x.min(), x.max(), y[y_limit], y.min()], aspect=aspect)
        # plt.yticks(np.arange(start=0, stop=y_limit, step=step), y[:y_limit:step])

    plt.ylabel('Distance[m]')
    plt.xlabel('velocity[m/s]')
    plt.colorbar()
    plt.show()

    return


def get_window(frames, start, length):
    return frames[start:start + length, :]


def get_window_from_spect(spect, start, length):
    return spect[:, start:start + length]


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


def get_spectogram_slices(spectrogram: np.ndarray, window_size: int = 216) -> np.ndarray:
    """
    splits spectogram into vertical slices of size spectrogram.shape[0] x window_size
    returns np.array of shape (num_slices, spectrogram.shape[0], window_size, [1]) (can add class_id)
    """
    stride = window_size // 2
    num_windows = spectrogram.shape[1] // stride - 1
    slices = np.empty((num_windows, spectrogram.shape[0], window_size))
    for x in range(num_windows):
        slices[x] = spectrogram[:, (x * stride):(x * stride) + window_size]

    return slices


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


def generate_kalman_trajectory(differential_spectrogram: np.ndarray,
                               y_cut: int = 90,
                               x_cut: int = 1000,
                               starting_q: float = 10e-20,
                               starting_r: float = 10e-5,
                               start_from_end: bool = True,
                               variance_window: int = 10,
                               variance_treshold: int = 100,
                               trash_q_value: float = 10e-25,
                               trash_r_value: float = 10e+5,
                               real_q_value: float = 10e-15,
                               real_r_value: float = 10e-2
                               ):
    """
    :param differential_spectrogram: spectrogram to predict trajectory for
    :param y_cut: cutoff in y-axis (default 90, which corresponds to 12 meters)
    :param x_cut: cutoff in x-axis (default 1 second from each side of the recording)
    :param start_from_end: whether to start the filter from end to beginning
    :param starting_q: important parameter #1, tweaking it costs sanity
    :param starting_r: important parameter #2, tweaking it costs sanity
    :param variance_window: important parameter #3, tweaking it costs sanity
    :param variance_treshold: important parameter #4, tweaking it costs sanity
    :param trash_q_value: important parameter #5, tweaking it costs sanity
    :param trash_r_value: important parameter #6, tweaking it costs sanity
    :param real_q_value: important parameter #7, tweaking it costs sanity
    :param real_r_value: important parameter #8, tweaking it costs sanity
    :return: predicted trajectory
    """

    if start_from_end:
        max_indices = np.argmax(differential_spectrogram[:y_cut, x_cut:-x_cut], axis=0)[::-1]
    else:
        max_indices = np.argmax(differential_spectrogram[:y_cut, x_cut:-x_cut], axis=0)

    input_data = max_indices[:, np.newaxis]

    kf = KalmanFilter(dim_x=2, dim_z=1)

    # Define the state transition matrix
    dt = 1.0  # time step
    kf.F = np.array([[1, dt], [0, 1]])

    # Define the measurement function matrix
    kf.H = np.array([[1, 0]])

    # Define the process noise covariance matrix
    q = starting_q  # process noise
    kf.Q = np.array([[q * dt ** 3 / 3, q * dt ** 2 / 2],
                     [q * dt ** 2 / 2, q * dt]])

    # Define the measurement noise covariance matrix
    r = starting_r  # measurement noise
    kf.R = np.array([[r]])

    # Define the initial state and covariance matrix
    x0 = np.array([max_indices[0], 0])  # initial state (position, velocity)
    p0 = np.eye(2)  # initial covariance matrix

    # Initialize the filter with the initial state and covariance matrix
    kf.x = x0
    kf.P = p0

    positions = np.zeros((max_indices.shape[0], 1))

    for t in range(max_indices.shape[0]):
        # Initialize the state estimate and error covariance matrix
        # Define the measurement as the current column of the data array
        current_var = np.var(max_indices[t - variance_window:t + variance_window])

        if current_var > variance_treshold:

            q = trash_q_value  # process noise
            kf.Q = np.array([[q * dt ** 3 / 3, q * dt ** 2 / 2],
                             [q * dt ** 2 / 2, q * dt]])
            r = trash_r_value  # measurement noise
            kf.R = np.array([[r]])

        else:
            q = real_q_value  # process noise
            kf.Q = np.array([[q * dt ** 3 / 3, q * dt ** 2 / 2],
                             [q * dt ** 2 / 2, q * dt]])

            r = real_r_value  # measurement noise
            kf.R = np.array([[r]])

        z = np.array([input_data[t]])  # measurement at time i
        kf.predict()
        kf.update(z)
        positions[t] = kf.x[0]

    if start_from_end:
        return np.flip(np.where(positions < y_cut, positions, y_cut), axis=0)
    else:
        return np.where(positions < y_cut, positions, y_cut)


def generate_multiple_kalman_trajectories(sample_file: np.ndarray,
                                          differential_values: list,
                                          y_cut: int = 90,
                                          x_cut: int = 1000,
                                          starting_q: float = 10e-20,
                                          starting_r: float = 10e-5,
                                          start_from_end: bool = True,
                                          variance_window: int = 10,
                                          variance_treshold: int = 100,
                                          trash_q_value: float = 10e-25,
                                          trash_r_value: float = 10e+5,
                                          real_q_value: float = 10e-15,
                                          real_r_value: float = 10e-2
                                          ) -> list:
    """
    :param sample_file: raw file to predict trajectory for
    :param differential_values: list of differences to predict trajectories for
    :param y_cut: cutoff in y-axis (default 90, which corresponds to 12 meters)
    :param x_cut: cutoff in x-axis (default 1 second from each side of the recording)
    :param start_from_end: whether to start the filter from end to beginning
    :param starting_q: important parameter #1, tweaking it costs sanity
    :param starting_r: important parameter #2, tweaking it costs sanity
    :param variance_window: important parameter #3, tweaking it costs sanity
    :param variance_treshold: important parameter #4, tweaking it costs sanity
    :param trash_q_value: important parameter #5, tweaking it costs sanity
    :param trash_r_value: important parameter #6, tweaking it costs sanity
    :param real_q_value: important parameter #7, tweaking it costs sanity
    :param real_r_value: important parameter #8, tweaking it costs sanity
    :return: list of predicted trajectories
    """

    vectors = []

    for differential_value in differential_values:
        frames_diff = diff_frames(sample_file, differential_value)
        diff_spect, y = gen_spectogram(frames_diff)
        diff_spectdb = to_dB(diff_spect)
        vectors.append(generate_kalman_trajectory(diff_spectdb, y_cut, x_cut, starting_q,
                                                  starting_r, start_from_end, variance_window, variance_treshold,
                                                  trash_q_value, trash_r_value, real_q_value, real_r_value))

    return vectors


def cut_trajectory_from_spectrogram(spectrogram_to_cut_from: np.ndarray,
                                    trajectory: np.ndarray,
                                    window_height: int = 4,
                                    y_cut: int = 90,
                                    x_cut: int = 1000
                                    ) -> np.ndarray:
    """
    :param spectrogram_to_cut_from: self-explanatory
    :param trajectory: trajectory in vector format (shape=X,)
    :param window_height: total height of the window counted from top to bottom
    :param y_cut: this parameter needs to be shared with generate_kalman_trajectory function,
                  otherwise there will be shape mismatch
    :param x_cut: this parameter needs to be shared with generate_kalman_trajectory function,
                  otherwise there will be shape mismatch
    :return: isolated trajectory from the general noise based on the trajectory vector values
    """
    modified_normal_spect = spectrogram_to_cut_from[:y_cut, x_cut:-x_cut]
    cut_trajectory = np.zeros_like(modified_normal_spect)

    for idx, column in enumerate(trajectory):
        window_up_limit = int(column[0]) - window_height // 2
        window_down_limit = int(column[0]) + window_height // 2
        window_up_limit, window_down_limit = _bounds_check(bounds=(window_up_limit, window_down_limit),
                                                           window_height=window_height)

        cut_trajectory[window_up_limit:window_down_limit, idx] = modified_normal_spect[
                                                                 window_up_limit:window_down_limit, idx]

    return cut_trajectory


def _bounds_check(bounds: tuple,
                  window_height: int) -> tuple:
    """
    :param bounds: bounds to check
    :param window_height: parameter shared with cut_trajectory_from_spectrogram function
    :return: legit bounds
    """
    window_up_limit, window_down_limit = bounds

    if window_up_limit < 0:
        window_up_limit = 0
        window_down_limit = (window_height // 2) * 2

    if window_down_limit > 90:
        window_up_limit = 90 - (window_height // 2) * 2
        window_down_limit = 90

    return window_up_limit, window_down_limit
