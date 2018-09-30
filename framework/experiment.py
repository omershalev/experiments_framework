import os
import datetime
import time
import pickle
import json
from collections import OrderedDict

import logger
import numpy as np

from utils import ExperimentFailure

_logger = logger.get_logger()


class Experiment(object):
    def __init__(self, name, data_sources, working_dir, params=None, metadata=None):
        self.name = name
        self.data_sources = data_sources
        self.working_dir = working_dir
        self.params = params
        self.metadata = metadata
        self.results = OrderedDict()
        self.valid_repetitions = []
        self.np_random_state = np.random.get_state()
        _logger.info('New experiment: %s' % self.name)

    def clean_env(self):
        raise NotImplementedError

    def prologue(self):
        pass

    def task(self, **kwargs):
        raise NotImplementedError

    def is_valid(self):
        return True

    def run(self, repetitions, **kwargs):
        try:
            experiment_dir_name = '%s_%s' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), self.name)
            self.experiment_dir = os.path.join(self.working_dir, experiment_dir_name)
            os.mkdir(self.experiment_dir)
            self.prologue()
            self.repetition_id = 1
            while self.repetition_id <= repetitions:
                self.results[self.repetition_id] = {}
                self.repetition_dir = os.path.join(self.experiment_dir, str(self.repetition_id))
                os.mkdir(self.repetition_dir)
                _logger.info('Beginning repetition #%d' % self.repetition_id)
                self.clean_env()
                self.task(**kwargs)
                if self.is_valid():
                    _logger.info('Valid repetition')
                    self.valid_repetitions.append(self.repetition_id)
                    self.repetition_id += 1
                else:
                    _logger.info('Invalid repetition')
                    os.rename(self.repetition_dir, '%s_invalid' % self.repetition_dir)
                    self.repetition_id += 1
            def make_serializable(value):
                if type(value) is dict:
                    for k, v in value.items():
                        value[k] = make_serializable(v)
                    return value
                if type(value) is np.ndarray:
                    return value.tolist()
                return value
            params_to_dump = {key: make_serializable(value) for key, value in self.params.items()} if self.params is not None else None
            metadata_to_dump = {key: make_serializable(value) for key, value in self.metadata.items()} if self.metadata is not None else None
            results_to_dump = {key: make_serializable(value) for key, value in self.results.items()}
            summary = {'name': self.name,
                       'data_sources': self.data_sources,
                       'params': params_to_dump,
                       'metadata': metadata_to_dump,
                       'results': results_to_dump,
                       'experiment_dir': self.experiment_dir,
                       'valid_repetitions': self.valid_repetitions}
            self.summary = summary
            with open(os.path.join(self.experiment_dir, 'experiment_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            with open(os.path.join(self.experiment_dir, 'numpy_random_state.pkl'), 'wb') as f:
                pickle.dump(self.np_random_state, f)
        except ExperimentFailure:
            _logger.info('Experiment failed, continuing execution')
        finally:
            self.clean_env()
            time.sleep(3)
        _logger.info('End of experiment: %s' % self.name)