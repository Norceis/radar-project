# %%
import logging
from functools import partial
from typing import Iterable, List, Tuple

import numpy as np

NUM_CHANNELS = 2
NUM_ANTENNAS = 4
FRAME_LENGTH = 28
CHIRP_LENGTH = 512
BYTES_SAMPLE = 2
FRAME_SIZE = FRAME_LENGTH * NUM_ANTENNAS * NUM_CHANNELS * CHIRP_LENGTH * BYTES_SAMPLE


def twos_to_int(twos):
    """16 bit two's complement int to int"""
    mask = 1 << 15
    negative = -1 * (twos & mask)
    positive = twos & (~mask)
    return negative + positive


ChirpCheckpoint = Tuple[np.ndarray, Tuple[int, int, int]]


def iter_chirp(
    filename: str,
    channels: int = NUM_CHANNELS,
    rx_antennas: int = NUM_ANTENNAS,
    chirp_length: int = CHIRP_LENGTH,
    previous_checkpoint: ChirpCheckpoint = None,
):
    if previous_checkpoint is not None:
        buffer, (antenna_idx, channel_idx, sample_idx) = previous_checkpoint
    else:
        buffer = np.zeros((channels, rx_antennas, chirp_length), dtype=np.int16)

        antenna_idx = 0
        channel_idx = 0
        sample_idx = 0

    with open(filename, "rb") as infile:
        for chunk in iter(
            partial(infile.read, BYTES_SAMPLE), b""
        ):  # 16 bit chunks/ samples
            val = int.from_bytes(chunk, "little")
            decoded = twos_to_int(val)
            buffer[channel_idx, antenna_idx, sample_idx] = decoded
            antenna_idx = (antenna_idx + 1) % rx_antennas
            if not antenna_idx:
                channel_idx = int(not channel_idx)
                if not channel_idx:
                    sample_idx = (sample_idx + 1) % chirp_length
                    if not sample_idx:
                        yield buffer
                        buffer = np.zeros(
                            (channels, rx_antennas, chirp_length), dtype=np.int16
                        )

    # run out of file, but not finished chirp
    if antenna_idx or channel_idx or sample_idx:
        logging.warning(
            "CHIRP EOF @ %dANT %dCH %d sample",
            antenna_idx,
            channel_idx,
            sample_idx,
        )
        return buffer, (antenna_idx, channel_idx, sample_idx)


FrameCheckpoint = Tuple[List[np.ndarray], ChirpCheckpoint]


def iter_frame(
    chirps: Iterable,
    frame_length: int = FRAME_LENGTH,
    previous_checkpoint: FrameCheckpoint = None,
):
    if previous_checkpoint:
        previous_buffer, _ = previous_checkpoint
        buffer = previous_buffer
    else:
        buffer = []

    for chirp in chirps:
        buffer.append(chirp)

        if len(buffer) == frame_length:
            yield np.stack(buffer)
            buffer = []

    # run out of file, finished chirp, but not finished frame
    if len(buffer):
        logging.warning("FRAME EOF @ %d chirp", len(buffer))
        return buffer, None

    # get return value of the generator
    try:
        chirp = next(chirps)
    except StopIteration as e:
        # run out of file, but not finished chirp in frame
        return buffer, e.value
    else:
        raise RuntimeError("Generator still had some data")


def yield_chunk(frames: Iterable, chunk_length: int = None):
    buffer: List[np.ndarray] = []
    for frame in frames:
        buffer.append(frame)
        if chunk_length and len(buffer) >= chunk_length:
            yield np.stack(buffer)
            buffer = []

    # get return value of the generator
    try:
        frame = next(frames)
    except StopIteration as e:
        # run out of file, but not finished frame or chirp
        logging.warning("CHUNK EOF @ %d frame", len(buffer))
        yield np.stack(buffer), e.value
        buffer = []
    else:
        raise RuntimeError("Generator still had some data")


def get_chunk(frames: Iterable, chunk_length: int = None):
    try:
        # get just the first chunk
        chunk = next(yield_chunk(frames, chunk_length))
        return chunk
    except StopIteration as e:
        raise RuntimeError("Generator had no data") from e


# %%
if __name__ == "__main__":
    FILENAME = "D:\DCA1000EVM_Pomiary\pomiar_90_0_2_Raw_0.bin"

    FRAME_LEN = 28 - 1
    FPS = 20

    data_tensor = get_chunk(
        iter_frame(iter_chirp(FILENAME), frame_length=FRAME_LEN),
        chunk_length=-1,  # FPS * 30
    )
    print(data_tensor.shape, " (frames, frame chirps, channels, antennas, samples)")
