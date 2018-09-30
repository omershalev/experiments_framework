import cv2
import math
from framework import cv_utils
from framework import viz_utils
from astar.astar import AStar

class PathPlan(AStar):
    def __init__(self, map_image):
        self.map_image = map_image

    def heuristic_cost_estimate(self, current, goal):
        (x1, y1) = current
        (x2, y2) = goal
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        return 1 # TODO: change

    def neighbors(self, node):
        curr_x, curr_y = node
        def is_free(x, y):
            if 0 <= x < self.map_image.shape[1] and 0 <= y < self.map_image.shape[0]:
                if self.map_image[y, x] == 0:
                    return True
            return False
        return [(x, y) for (x, y) in [(curr_x, curr_y - 1), (curr_x, curr_y + 1),
                                      (curr_x - 1, curr_y), (curr_x + 1, curr_y)] if is_free(x,y)]


if __name__ == '__main__':
    map_image = cv2.cvtColor(cv2.imread(r'/home/omer/Downloads/dji_15-53-1_map.pgm'), cv2.COLOR_RGB2GRAY)
    points = cv_utils.sample_pixel_coordinates(map_image, multiple=True)
    start = points[0]
    goal = points[1]
    path_plan = PathPlan(map_image)
    path = path_plan.astar(start, goal)
    for point in path:
        cv2.circle(map_image, point, radius=3, color=255, thickness=-1)
    viz_utils.show_image('path', map_image)
    print ('end')