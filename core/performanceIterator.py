import logging
import time
from collections.abc import Iterable
import torch

class PerformanceIterator:
    def __init__(self, data_loader, checkpoint_dir, load_checkpoint_func,
            save_checkpoint_func, log_file, onGPU=False):
        if not isinstance(data_loader, Iterable):
            raise ValueError('Data is of uniterable '
                             'type %s' % (type(data_loader)))
        else:
            self._data_loader = data_loader
        self._load_checkpoint_func = load_checkpoint_func
        self._save_checkpoint_func = save_checkpoint_func
        self._checkpoint_dir = checkpoint_dir
        self._log_file = log_file
        self._onGPU = onGPU

        self._logger = logging.getLogger()
        logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG, filemode='w')
        
        self._steps = 0
        self._duration = 0
        self._done = False
        self._prev_time = None

        self._logger.info('[{0}][{1}][][]'.format(time.time(),'INIT'))

    def __del__(self):
        self.complete()

    def __iter__(self):
        self._iterator = iter(self._data_loader)
        return self

    def _record_step(self):
        """Helper to synchronize GPU and log elapsed compute time."""
        if self._onGPU:
            torch.cuda.synchronize()

        if self._prev_time is not None:
            elapsed_time = time.time() - self._prev_time
            self._duration += elapsed_time
            self._steps += 1
            self._write_info()

    def __next__(self):
        self._record_step()
        try:
            val = next(self._iterator)
        except StopIteration as e:
            self._done = True
            self.complete()
            raise StopIteration

        # Capture the time when the model STARTS its next compute loop, 
        # which is right after the iterator returns the next batch.
        self._prev_time = time.time()
        return val

    def __len__(self):
        return len(self._data_loader)

    @property
    def done(self):
        return self._done

    def complete(self):
        if not self._done:
            self._record_step()
            self._done = True
        self._logger.info('[{0}][{1}][{2}][{3}]'.format(time.time(),'COMPLETE','STEPS', self._steps))

    def _write_info(self):
        self._logger.info('[{0}][{1}][{2}][{3}]'.format(time.time(),'PROGRESS','STEPS', self._steps))
        self._logger.info('[{0}][{1}][{2}][{3}]'.format(time.time(),'PROGRESS','DURATION', self._duration))