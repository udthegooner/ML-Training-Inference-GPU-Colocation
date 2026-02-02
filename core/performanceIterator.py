import logging
import time
from collections.abc import Iterable

class PerformanceIterator:
    def __init__(self, data_loader, checkpoint_dir, load_checkpoint_func,
            save_checkpoint_func, log_file):
        if not isinstance(data_loader, Iterable):
            raise ValueError('Data is of uniterable '
                             'type %s' % (type(data_loader)))
        else:
            self._data_loader = data_loader
        self._load_checkpoint_func = load_checkpoint_func
        self._save_checkpoint_func = save_checkpoint_func
        self._checkpoint_dir = checkpoint_dir
        self._log_file = log_file
        self._logger = logging.getLogger()
        logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG, filemode='w')
        self._steps = 0
        self._duration = 0
        self._done = False
        self._write_info()
        self._prev_time = None

    def __del__(self):
        self.complete()

    def __iter__(self):
        self._iterator = iter(self._data_loader)
        return self

    def __next__(self):
        # Update the elapsed time.
        cur_time = time.time()
        if self._prev_time is None:
            self._prev_time = cur_time
        elapsed_time = cur_time - self._prev_time
        self._duration += elapsed_time
        self._prev_time = cur_time

        try:
            val = next(self._iterator)
            self._steps += 1
            self._write_info()
        except StopIteration as e:
            self._write_info()
            raise StopIteration

        return val

    def __len__(self):
        return len(self._data_loader)

    @property
    def done(self):
        return self._done

    def complete(self):
        self._done = True
        self._write_info()
        self._logger.info('[{0}][{1}][{2}][]'.format(time.time(),'PROGRESS',
            'COMPLETE'))

    def _write_info(self):
        self._logger.info('[{0}][{1}][{2}][{3}]'.format(time.time(),'PROGRESS',
            'STEPS', self._steps))
        self._logger.info('[{0}][{1}][{2}][{3}]'.format(time.time(),'PROGRESS',
            'DURATION', self._duration))
