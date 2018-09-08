import os
import subprocess
import functools
import multiprocessing
from threading import Timer
from joblib import Parallel, delayed
import datetime

import config


def run_timeout_protected_process(command, timeout_in_seconds):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = Timer(timeout_in_seconds, lambda proc: proc.kill(), [proc])
    try:
        timer.start()
        while proc.poll() is None:
            pass
        output = proc.stdout.read() if proc.stdout is not None else None
        error = proc.stderr.read() if proc.stderr is not None else None
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, ' '.join(command), output)
    finally:
        timer.cancel()
    return output, error


def new_process(command, cwd=None, output_to_console=None):
    if output_to_console is None:
        output_to_console = config.output_to_console
    if output_to_console:
        proc = subprocess.Popen(command, cwd=cwd)
    else:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    return proc


def kill_process(process_name):
    proc = new_process(['killall', '-9', process_name], output_to_console=False)
    proc.communicate()


def create_new_execution_folder(name):
    execution_dir_name = '%s_%s' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), name)
    execution_dir = os.path.join(config.base_output_path, execution_dir_name)
    os.mkdir(execution_dir)
    return execution_dir


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def joblib_map(func, iterable):
    return Parallel(
        n_jobs=-1,
        timeout=36000 # 10 hours
    )(delayed(func)(*args) for args in iterable)


def slice_handler(slice_start, slice_stop, iterable_to_split, func):
    ret_values = []
    for idx in xrange(slice_start, slice_stop):
        ret_values.append(func(iterable_to_split[idx]))
    return ret_values


def distribute_evenly_on_all_cores(func, iterable_to_split):
    split_step = len(iterable_to_split) / multiprocessing.cpu_count()
    slice_indices = range(0, len(iterable_to_split), split_step)
    slice_indices[-1] = len(iterable_to_split)
    slice_start_stop_tuples = [(slice_start, slice_stop) for slice_start, slice_stop in zip(slice_indices, slice_indices[1:])]
    joblib_ret_values = joblib_map(slice_handler, [(slice_start, slice_stop, iterable_to_split, func) for slice_start, slice_stop in slice_start_stop_tuples])
    ret = []
    for joblib_ret_value in joblib_ret_values:
        ret += joblib_ret_value
    return ret
