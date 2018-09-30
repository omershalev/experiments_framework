import os
import json
import numpy as np
base_path = r'/home/omer/orchards_ws/results/trunks_detection'

if __name__ == '__main__':
    dimx = []
    dimy = []
    for dirname in os.listdir(base_path):
        json_path = os.path.join(base_path, dirname, 'experiment_summary.json')
        if not os.path.exists(json_path):
            print 'skipping %s' % dirname
            continue
        with open(json_path) as f:
            summary = json.load(f)
        for repetition in summary['results']:
            dimx.append(summary['results'][str(repetition)]['optimized_grid_dim_x'])
            dimy.append(summary['results'][str(repetition)]['optimized_grid_dim_y'])
    mean_dimx = np.mean(dimx)
    mean_dimy = np.mean(dimy)
    print 'dim x: %s [%s]' % (mean_dimx, np.std(dimx))
    print 'dim y: %s [%s]' % (mean_dimy, np.std(dimy))

    measured_row_widths = [7.3, 7.15, 6.9, 7.2, 6.9]
    measured_intra_row_distances = [6.0, 5.7, 5.9, 5.9, 5.8, 5.85]

    width_mean = np.mean(measured_row_widths)
    intra_row_mean = np.mean(measured_intra_row_distances)
    print 'width: %s' % width_mean
    print 'intra-row: %s' % intra_row_mean

    print ''

    print 'dim x / width: %f' % (float(mean_dimx) / float(width_mean))
    print 'dim y / intra-row: %f' % (float(mean_dimy) / float(intra_row_mean))

    print '====='

    print 'dim y / width: %f' % (float(mean_dimy) / float(width_mean))
    print 'dim x / intra-row: %f' % (float(mean_dimx) / float(intra_row_mean))