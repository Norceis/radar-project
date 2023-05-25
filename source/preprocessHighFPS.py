import os
import logging
from pathlib import Path

from datetime import datetime

import numpy as np
from scipy import fft
# from source import zfft

from source.DCA1000EVMreader import (
    ChirpCheckpoint,
    FrameCheckpoint,
    yield_chunk,
    get_chunk,
    iter_chirp,
    iter_frame,
)


def aggregate_bin_files(
    filename: str,
    frame_length: int = 3,
    limit: int = 10,
    radar_dir: Path = (Path("/DCA1000EVM/Pomiary_HighFPS/") / "radar"),
):
    radar_files = _get_radar_filenames(filename, radar_dir)
    radar_files = radar_files[: (limit - 1)]

    # load each file into memory
    data_tensor = []
    chirp_checkpoint: ChirpCheckpoint = None
    frame_checkpoint: FrameCheckpoint = None
    for radar_file in radar_files:
        chunk = _load_file(frame_length, chirp_checkpoint, frame_checkpoint, radar_file)

        data_tensor.append(chunk)

    # aggregate all data
    return np.concatenate(data_tensor)


def yield_aggregate_bin_files(
    filename: str,
    frame_length: int = 3,
    chunk_length: int = 1000 * 10,
    file_limit: int = None,
    radar_dir: Path = (Path("/DCA1000EVM/Pomiary_HighFPS/") / "radar"),
):
    radar_files = _get_radar_filenames(filename, radar_dir)
    if isinstance(file_limit, int):
        radar_files = radar_files[: (file_limit - 1)]

    data_tensor = []
    data_tensor_length = 0
    chirp_checkpoint: ChirpCheckpoint = None
    frame_checkpoint: FrameCheckpoint = None
    for radar_file in radar_files:
        logging.warning(radar_file)

        frame_iterator = iter_frame(
            iter_chirp(radar_file, previous_checkpoint=chirp_checkpoint),
            frame_length=frame_length,
            previous_checkpoint=frame_checkpoint,
        )

        for chunk in yield_chunk(
            frame_iterator,
            chunk_length=chunk_length - data_tensor_length,
        ):
            if not isinstance(chunk, np.ndarray):
                chunk, frame_checkpoint = chunk
                if frame_checkpoint is not None:
                    _, chirp_checkpoint = frame_checkpoint

            data_tensor.append(chunk)
            data_tensor_length += len(chunk)

            if data_tensor_length >= chunk_length:
                yield np.concatenate(data_tensor)
                data_tensor = []
                data_tensor_length = 0

def yield_aggregate_bin_files_new(
    filename: str,
    frame_length: int = 3,
    chunk_length: int = 1000 * 10,
    file_limit: int = None,
    radar_dir: Path = (Path("/DCA1000EVM/Pomiary_HighFPS/") / "radar"),
):
    radar_files = _get_radar_filenames_new(filename, radar_dir)
    if isinstance(file_limit, int):
        radar_files = radar_files[: (file_limit - 1)]

    data_tensor = []
    data_tensor_length = 0
    chirp_checkpoint: ChirpCheckpoint = None
    frame_checkpoint: FrameCheckpoint = None
    for radar_file in radar_files:
        logging.warning(radar_file)

        frame_iterator = iter_frame(
            iter_chirp(radar_file, previous_checkpoint=chirp_checkpoint),
            frame_length=frame_length,
            previous_checkpoint=frame_checkpoint,
        )

        for chunk in yield_chunk(
            frame_iterator,
            chunk_length=chunk_length - data_tensor_length,
        ):
            if not isinstance(chunk, np.ndarray):
                chunk, frame_checkpoint = chunk
                if frame_checkpoint is not None:
                    _, chirp_checkpoint = frame_checkpoint

            data_tensor.append(chunk)
            data_tensor_length += len(chunk)

            if data_tensor_length >= chunk_length:
                yield np.concatenate(data_tensor)
                data_tensor = []
                data_tensor_length = 0


def _bin_file_index(filename: Path) -> int:
    return int(filename.name.split("_Raw_")[-1].replace(".bin", ""))


def _get_radar_filenames(filename, radar_dir):
    radar_files = sorted(list(radar_dir.glob(filename + "_*.bin")), key=_bin_file_index)
    return radar_files

def _get_radar_filenames_new(filename, radar_dir):
    radar_files = sorted(list(radar_dir.glob(filename + ".bin")), key=_bin_file_index)
    return radar_files


def _load_file(frame_length, chirp_checkpoint, frame_checkpoint, radar_file):
    chunk = get_chunk(
        iter_frame(
            iter_chirp(radar_file, previous_checkpoint=chirp_checkpoint),
            frame_length=frame_length,
            previous_checkpoint=frame_checkpoint,
        )
    )
    if not isinstance(chunk, np.ndarray):
        chunk, frame_checkpoint = chunk
        if frame_checkpoint is not None:
            _, chirp_checkpoint = frame_checkpoint
    return chunk


def _to_analytic(data_tensor):
    # convert signal to complex representation (this doubles the storage requirements)
    return data_tensor[:, :, 0, :, :] + 1j * data_tensor[:, :, 1, :, :]


def _chunk_to_fft(chunk, fft_size: int = None):
    # chunk : (chunk_length, TX_antenna, RX_antenna, chirp_length)
    chunk = np.asarray(chunk)

    chirp_len = chunk.shape[-1]

    window = np.hamming(chirp_len)
    chunk_windowed = chunk * window

    if fft_size is None:
        fft_size = chirp_len

    spectrum = fft.fft(chunk_windowed, n=fft_size, axis=-1)
    return spectrum


def _chunk_to_zfft(
    chunk,
    fft_size: int = None,
    freq_low: float = 0.0,
    freq_high: float = 100.0,
):
    # TODO: verify correctness of this code
    # chunk : (chunk_length, TX_antenna, RX_antenna, chirp_length)
    chunk = np.asarray(chunk)

    chirp_len = chunk.shape[-1]

    window = np.hamming(chirp_len)
    chunk_windowed = chunk * window

    if fft_size is None:
        fft_size = chirp_len

    spectrum = zfft.zfft(
        chunk_windowed,
        f0=freq_low,
        f1=freq_high,
        M=fft_size,
        axis=-1,
    )
    return spectrum


def _cache_file_name(
    filename: str,
    radar_dir: Path = (Path("/DCA1000EVM/Pomiary_HighFPS/") / "radar"),
):
    return radar_dir / (filename + "_cache.npy")


def _from_cache(
    filename: str,
    radar_dir: Path = (Path("/DCA1000EVM/Pomiary_HighFPS/") / "radar"),
):
    cache_file = _cache_file_name(filename, radar_dir)

    if os.path.exists(cache_file):
        return np.load(cache_file, mmap_mode="r")
    else:
        return None


def _to_cache(
    signal: np.ndarray,
    filename: str,
    radar_dir: Path = (Path("/DCA1000EVM/Pomiary_HighFPS/") / "radar"),
):
    cache_file = _cache_file_name(filename, radar_dir)

    np.save(cache_file, signal)

    return np.load(cache_file, mmap_mode="r")


def _get_metadata(
    filename: str,
    radar_dir: Path = (Path("/DCA1000EVM/Pomiary_HighFPS/") / "radar"),
):
    with open(radar_dir / (filename + "_LogFile.csv")) as metafile:
        for line in metafile:
            if line.startswith("Capture start time"):
                _, timestamp = line.split("-")
                timestamp = timestamp.strip()
                _, month, day, time, year = timestamp.split()
                month = datetime.strptime(month, "%b").month
                day = int(day)
                year = int(year)
                time = [int(val) for val in time.split(":")]
                start_timestamp = datetime(year, month, day, *time)  # tz = CEST
            if line.startswith("Duration(sec)"):
                _, duration = line.split("-")
                duration = duration.strip()
                duration = int(duration)

    return start_timestamp, duration


def _find_zephyr_file(
    start_timestamp: datetime,
    zephyr_dir: Path = (Path("/DCA1000EVM/Pomiary_HighFPS/") / "zephyr"),
):
    zephyr_files = list(zephyr_dir.glob("*_BR_RR.csv"))
    zephyr_times = [
        datetime.fromisoformat(
            filepath.name.replace("_BR_RR.csv", "").split("__")[0].replace("_", "-")
            + "T"
            + filepath.name.replace("_BR_RR.csv", "").split("__")[-1].replace("_", ":")
        )
        for filepath in zephyr_files
    ]
    zephyr_delta = [
        abs((timestamp - (start_timestamp)).total_seconds())
        for timestamp in zephyr_times
    ]
    min_delta = min(zephyr_delta)
    min_index = zephyr_delta.index(min_delta)
    zephyr_filename = zephyr_files[min_index].name
    return zephyr_dir / zephyr_filename


if __name__ == "__main__":

    BASE_DIR = Path("/DCA1000EVM/Pomiary_HighFPS/")
    RADAR_DIR = BASE_DIR / "radar"

    for radar_filename, frame_len in [
        # ("xxx_0_0_randomSitWalkMinimalMultiRXMultiTX_2022-08-05_13-19-00_Raw", 3),
        ("500_0_0_stand_2022-08-05_13-35-00_Raw", 3),
        ("200_0_0_walk1mrad_2022-07-27_19-44-00_Raw", 3),
        ("200_0_0_randomWalk_2022-07-29_20-22-00_Raw", 3),
        ("200_0_0_sit_2022-07-27_21-18-00_Raw", 3),
        ("300_0_0_sit_2022-07-29_20-01-00_Raw", 3),
        ("200_0_0_sit_2022-07-29_18-42-00_Raw", 3),
        ########################################################################
        ("xxx_0_0_randomSitWalkMinimalMultiRX_2022-08-05_13-02-00_Raw", 1),
        ("500_0_0_standSingleTx_2022-08-05_13-52-00_Raw", 1),
        ########################################################################
        # ("500_0_0_standSingleTxSingleRx_2022-08-05_14-03-00_Raw", 1),
        # ("xxx_0_0_randomSitWalkMinimal_2022-08-05_12-46-00_Raw", 1),
    ]:
        # if _from_cache(radar_filename) is not None:
        #     continue
        # print(radar_filename)

        # data_tensor = aggregate_bin_files(radar_filename, frame_length, limit=3)

        # analytic_signal = _to_analytic(data_tensor)

        # data_tensor = data_tensor[:, :1, :, :1, :]  # only RX0 and TX0 have valid data
        # analytic_signal = _to_cache(analytic_signal, radar_filename)

        for data_chunk in yield_aggregate_bin_files(
            radar_filename,
            frame_len,
            chunk_length=4242,
            # chunk_length=43690 + 5000,  # a little bit more than one file
        ):
            print(data_chunk.shape)
            analytic_chunk = _to_analytic(data_chunk)
            print(analytic_chunk.shape)
