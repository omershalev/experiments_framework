import os
import datetime
import pickle
import hashlib
from collections import OrderedDict

import logger
import utils

class Experiment(object):
    def __init__(self, name, data_sources, working_dir, params=None):
        self.name = name
        self.data_sources = data_sources
        self.working_dir = working_dir
        self.params = params
        self.results = OrderedDict()
        self._calculate_md5()
        self._logger = logger.get_logger()

    def _calculate_md5(self):
        if type(self.data_sources) == tuple:
            hash_obj = hashlib.md5(open(self.data_sources[0], 'rb').read())
            for data_source in self.data_sources:
                hash_obj.update(open(data_source, 'rb').read())
                self._md5 = hash_obj.hexdigest()
        else:
            self._md5 = hashlib.md5(open(self.data_sources, 'rb').read()).hexdigest()

    def task(self):
        raise NotImplementedError

    def is_valid(self):
        return True

    def run(self, repetitions):
        experiment_dir_name = '%s_%s' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), self.name)
        self.experiment_dir = os.path.join(self.working_dir, experiment_dir_name)
        os.mkdir(self.experiment_dir)
        i = 0
        while i < repetitions:
            self.repetition_dir = os.path.join(self.experiment_dir, str(i))
            os.mkdir(self.repetition_dir)
            self._logger.info('Beginning repetition #%d' % i)
            try:
                utils.ros_kill_all()
                self.task() # TODO: on theory, this should be timeout protected...
                if self.is_valid():
                    self._logger.info('Valid execution')
                    i += 1
                else:
                    self._logger.info('Invalid execution')
            except Exception as e:
                self._logger.error('Encountered an error: %s' % e)
        # with open(os.path.join(self.experiment_dir, self.name), 'wb') as out_file:
        #     pickle.dump(self, out_file)