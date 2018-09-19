import os
import datetime
import pickle
import hashlib
import json
from collections import OrderedDict

import logger
import numpy as np

_logger = logger.get_logger()


class Experiment(object):
    def __init__(self, name, data_sources, working_dir, params=None):
        self.name = name
        self.data_sources = data_sources
        self.working_dir = working_dir
        self.params = params
        self.results = OrderedDict()
        _logger.info('New experiment: %s' % self.name)

    def clean_env(self):
        raise NotImplementedError

    def task(self, **kwargs):
        raise NotImplementedError

    def is_valid(self):
        return True

    def run(self, repetitions, **kwargs):
        try:
            experiment_dir_name = '%s_%s' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), self.name)
            self.experiment_dir = os.path.join(self.working_dir, experiment_dir_name)
            os.mkdir(self.experiment_dir)
            self.repetition_id = 1
            while self.repetition_id <= repetitions:
                self.results[self.repetition_id] = {}
                self.repetition_dir = os.path.join(self.experiment_dir, str(self.repetition_id))
                os.mkdir(self.repetition_dir)
                _logger.info('Beginning repetition #%d' % self.repetition_id)
                self.clean_env()
                self.task(**kwargs)
                if self.is_valid():
                    _logger.info('Valid execution')
                    self.repetition_id += 1
                else:
                    _logger.info('Invalid execution')
                    os.rename(self.repetition_dir, '%s_invalid' % self.repetition_dir)
                    self.repetition_id += 1
            def make_serializable(value):
                if type(value) is np.ndarray:
                    return value.tolist()
                return value
            params_to_dump = {key: make_serializable(value) for key, value in self.params.items()}
            results_to_dump = {key: make_serializable(value) for key, value in self.results.items()}
            json_obj = {'name': self.name,
                        'data_sources': self.data_sources,
                        'params': params_to_dump,
                        'results': results_to_dump}
            with open(os.path.join(self.experiment_dir, 'experiment_metadata.json'), 'w') as json_file:
                json.dump(json_obj, json_file, indent=4)
        finally:
            self.clean_env()