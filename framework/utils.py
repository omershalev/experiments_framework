import os
import subprocess
import functools
from threading import Timer
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


def create_new_execution_folder(name):
    execution_dir_name = '%s_%s' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), name)
    execution_dir = os.path.join(config.base_output_path, execution_dir_name)
    os.mkdir(execution_dir)
    return execution_dir


def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))