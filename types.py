from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass

import numpy as np


class UiCoordData:

    def __init__(self, jobs_shm_name: str, job_id: int):
        self.jobs_shm_name = jobs_shm_name
        self.job_id = job_id
        self.__shm: SharedMemory | None = None

    def copy(self):
        """
        @return: a copy with the same jobs_shm_name & job_id only; no shm_data.
        """
        return UiCoordData(self.jobs_shm_name, self.job_id)

    @property
    def _shm(self):
        if self.__shm is None:
            self.__shm = SharedMemory(name=self.jobs_shm_name)
        return self.__shm

    def add_to_job_data_slot_and_check_interrupt(self, to_increment: int | np.uint32):
        shm_data_array = np.ndarray((2 + self.job_id,), dtype=np.dtype('uint32'), buffer=self._shm.buf)
        shm_data_array[1 + self.job_id] += to_increment
        return shm_data_array[0] > 0

