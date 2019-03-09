import numpy as np

plot_pattern = np.array([[ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
    'narrow_row': [('8/I', '9/I'),
                   ('8/H', '9/H'),
                   ('8/G', '9/G'),
                   ('8/F', '9/F'),
                   ('8/E', '9/E'),
                   ('8/D', '9/D'),
                   ('8/C', '9/C'),
                   ('8/B', '9/B'),
                   ('8/A', '9/A')],
    'wide_row': [('7/I', '8/I'), ('7/A', '8/A')],
    'u_turns': [('2/C', '3/C'),
                ('2/A', '3/A', 0, 150, 0, 150),
                ('3/A', '4/A', 0, 150, 0, 150),
                ('3/C', '4/C'),
                ('3/A', '4/A', 0, 150, 0, 150),
                ('5/A', '6/A', 0, 150, 0, 150),
                ('5/C', '6/C'),
                ('5/A', '6/A', 0, 150, 0, 150),
                ('7/A', '8/A', 0, 150, 0, 150),
                ('7/C', '8/C'),
                ('7/A', '8/A', 0, 150, 0, 150),
                ('8/A', '9/A', 0, 150, 0, 150),
                ('8/C', '9/C')],
    's_patrol': [('3/G', '4/G'), ('3/E', '4/E'), ('3/C', '4/C'),
                 ('4/C', '5/C'), ('4/E', '5/E'), ('4/G', '5/G'),
                 ('5/G', '6/G'), ('5/E', '6/E'), ('5/C', '6/C'),
                 ('6/C', '7/C'), ('6/E', '7/E'), ('6/G', '7/G'),
                 ('7/G', '8/G'), ('7/E', '8/E'), ('7/C', '8/C')],
    'tasks_and_interrupts': [('1/D', '2/D'),
                             ('3/C', '4/C'),
                             ('3/F', '4/F'),
                             ('5/G', '6/G'),
                             ('5/B', '6/B'),
                             ('7/A', '8/A'),
                             ('4/C', '5/C'),
                             ('7/C', '8/C'),
                             ('8/I', '9/I')]
}