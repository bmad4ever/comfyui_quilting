from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np


num_pixels: TypeAlias = int
percentage: TypeAlias = float


class UiCoordData:

    def __init__(self, jobs_shm_name: str, job_id: int):
        self.jobs_shm_name = jobs_shm_name
        self.job_id = job_id
        self.__shm: SharedMemory | None = None

    @property
    def _shm(self):
        if self.__shm is None:
            self.__shm = SharedMemory(name=self.jobs_shm_name)
        return self.__shm

    def add_to_job_data_slot_and_check_interrupt(self, to_increment: int | np.uint32):
        shm_data_array = np.ndarray((2 + self.job_id,), dtype=np.dtype('uint32'), buffer=self._shm.buf)
        shm_data_array[1 + self.job_id] += to_increment
        return shm_data_array[0] > 0


@dataclass
class GenParams:
    """
    Data used across multiple quilting subroutines.
    Used in quilting.py and make_seamless.py
    """
    block_size: num_pixels
    overlap: num_pixels
    tolerance: percentage
    blend_into_patch: bool
    version: int

    @property
    def bo(self):
        return self.block_size, self.overlap

    @property
    def bot(self):
        return self.block_size, self.overlap, self.tolerance
