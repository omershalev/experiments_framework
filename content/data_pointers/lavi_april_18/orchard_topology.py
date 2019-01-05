import numpy as np

plot_pattern = np.array([[-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8)

measured_row_widths = [7.3, 7.15, 6.9, 7.2, 6.9]
measured_intra_row_distances = [6.0, 5.7, 5.9, 5.9, 5.8, 5.85]
measured_trunks_perimeters = [0.75, 0.80, 0.85, 0.87, 0.84, 0.8, 0.86, 0.72, 0.83, 0.80]

trajectories = {
    # 'short_for_demo': [('8/I', '9/I'), ('8/G', '9/G')], # TODO: remove
    'narrow_row': [('8/I', '9/I'), ('8/A', '9/A')],
    # 'wide_row': [('7/I', '8/I'), ('7/A', '8/A')],# TODO: comment in
    # 'between_plots': [('1/A', '2/A'), ('9/A', '10/A')],
    # 'u_turns': [('2/C', '3/C'), ('2/A', '3/A'),
    #            ('3/A', '4/A'), ('3/C', '4/C'), ('3/A', '4/A'),
    #            ('5/A', '6/A'), ('5/C', '6/C'), ('5/A', '6/A'),
    #            ('7/A', '8/A'), ('7/C', '8/C'), ('7/A', '8/A'),
    #            ('8/A', '9/A'), ('8/C', '9/C')],
    # 's_patrol': [('4/A', '5/A'), ('4/I', '5/I'), ('5/I', '6/I'), ('5/A', '6/A'), ('6/A', '7/A'), ('6/I', '7/I')],
    # 'tasks_and_interrupts': ['1/D', '3/C', '3/F', '5/G', '5/B', '7/A', '4/C', '7/C']
}