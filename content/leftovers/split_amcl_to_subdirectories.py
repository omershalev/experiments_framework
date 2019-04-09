import os
import json
import shutil

amcl_execution_dir = r'/home/omer/Desktop/apr_noises_selected_instances'
output_dir = r'/home/omer/Desktop/apr_noises_selected_instances_processed'


if __name__ == '__main__':
    scan_noise_experiments_mapping = {0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: []}
    odometry_noise_sigma_x_experiments_mapping = {0.01: [], 0.02: [], 0.03: [], 0.04: [], 0.05: []}
    for experiment_name in os.listdir(amcl_execution_dir):
        with open(os.path.join(amcl_execution_dir, experiment_name, 'experiment_summary.json')) as f:
            summary = json.load(f)
        scan_noise_sigma = summary['params']['scan_noise_sigma']
        odometry_noise_sigma_x = summary['params']['odometry_noise_sigma_x']
        if scan_noise_sigma != 0:
            scan_noise_experiments_mapping[scan_noise_sigma].append(experiment_name)
        if odometry_noise_sigma_x != 0:
            odometry_noise_sigma_x_experiments_mapping[odometry_noise_sigma_x].append(experiment_name)
    for scan_noise_sigma, experiment_names in scan_noise_experiments_mapping.items():
        os.mkdir(os.path.join(output_dir, 'scan_noise=%f' % scan_noise_sigma))
        for experiment_name in experiment_names:
            shutil.copytree(os.path.join(amcl_execution_dir, experiment_name), os.path.join(output_dir, 'scan_noise=%f' % scan_noise_sigma, experiment_name))
            shutil.copy(os.path.join(amcl_execution_dir, experiment_name, 'error.png'), os.path.join(output_dir, 'scan_noise=%f' % scan_noise_sigma, 'error_%s.png' % experiment_name))
    for odometry_noise_sigma_x, experiment_names in odometry_noise_sigma_x_experiments_mapping.items():
        os.mkdir(os.path.join(output_dir, 'odometry_sigma_x=%f' % odometry_noise_sigma_x))
        for experiment_name in experiment_names:
            shutil.copytree(os.path.join(amcl_execution_dir, experiment_name), os.path.join(output_dir, 'odometry_sigma_x=%f' % odometry_noise_sigma_x, experiment_name))
            shutil.copy(os.path.join(amcl_execution_dir, experiment_name, 'error.png'), os.path.join(output_dir, 'odometry_sigma_x=%f' % odometry_noise_sigma_x, 'error_%s.png' % experiment_name))
    print ('end')

