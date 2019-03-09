import os
import pandas as pd
import matplotlib.pyplot as plt
save_path = r'/home/omer/Downloads/graphs'


def plot_canopies_vs_trunks(plot_name, canopies_vector, trunks_vector, canopies_stds, trunks_stds, output_dir):
    plt.figure()
    def plot_line_with_sleeve(vector, sleeve_width, color):
        upper_bound = vector + sleeve_width / 2
        lower_bound = vector - sleeve_width / 2
        plt.plot(vector, {'sienna': 'sienna', 'green': 'green', 'brown': 'brown'}[color])
        plt.plot(upper_bound, '%s' % {'sienna': 'sienna', 'green': 'green', 'brown': 'brown'}[color], linestyle='--')
        plt.plot(lower_bound, '%s' % {'sienna': 'sienna', 'green': 'green', 'brown': 'brown'}[color], linestyle='--')
        plt.fill_between(vector.index, upper_bound, lower_bound, facecolor=color, alpha=0.2)
    plot_line_with_sleeve(canopies_vector, canopies_stds, 'green')
    plot_line_with_sleeve(trunks_vector, trunks_stds, 'sienna')
    # plt.xlim((0.8, result_samples_num + 0.2))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '%s.png' % plot_name))

if __name__ == '__main__':
    canopies_mean_errors = pd.read_pickle(os.path.join(save_path, 'canopies_mean_errors.pkl'))
    canopies_std_errors = pd.read_pickle(os.path.join(save_path, 'canopies_std_errors.pkl'))
    canopies_mean_covariance_norms = pd.read_pickle(os.path.join(save_path, 'canopies_mean_covariance_norms.pkl'))
    canopies_std_covariance_norms = pd.read_pickle(os.path.join(save_path, 'canopies_std_covariance_norms.pkl'))
    trunks_mean_errors = pd.read_pickle(os.path.join(save_path, 'trunks_mean_errors.pkl'))
    trunks_std_errors = pd.read_pickle(os.path.join(save_path, 'trunks_std_errors.pkl'))
    trunks_mean_covariance_norms = pd.read_pickle(os.path.join(save_path, 'trunks_mean_covariance_norms.pkl'))
    trunks_std_covariance_norms = pd.read_pickle(os.path.join(save_path, 'trunks_std_covariance_norms.pkl'))
    plot_canopies_vs_trunks('omer', canopies_mean_errors, trunks_mean_errors, canopies_std_errors, trunks_std_errors, save_path)