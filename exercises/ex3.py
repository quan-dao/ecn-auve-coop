import numpy as np
from einops import rearrange
from exercises.utils import *
from armen_v2x.dataset.v2x_sim.v2x_sim_utils import get_annotated_boxes_in_sensor_frame, get_available_point_clouds, \
    CLASS_COLORS
from armen_v2x.utils.visualization import show_point_cloud
from armen_v2x.utils.geometry import find_points_in_boxes


def main():
    nusc, sample_token = get_a_sample()
    sample_rec = nusc.get('sample', sample_token)
    ref_sensor_name = 'LIDAR_TOP_id_1'
    points, points_src_idx = get_available_point_clouds(nusc, sample_token, ref_sensor_name,
                                                        THRESHOLD_DISTANCE_TO_LIDAR)
    # points: (N, 3[+C])

    boxes = get_annotated_boxes_in_sensor_frame(nusc, sample_rec['data'][ref_sensor_name])
    box_colors = None  # TODO

    boxes_to_points = find_points_in_boxes(points, boxes)  # (N,)

    print('showing foreground points (class-agnostic)')
    points_color_by_fg = np.zeros((points.shape[0], 3))
    points_color_by_fg[boxes_to_points > -1] = np.array([1, 0, 0])  # red for foreground points
    show_point_cloud(points[:, :3], boxes[:, :7], points_color_by_fg, box_colors)

    print('showing foreground points (class-aware)')
    # TODO: compute points_color_by_cls
    points_color_by_cls = None
    show_point_cloud(points[:, :3], boxes[:, :7], points_color_by_cls, box_colors)


if __name__ == '__main__':
    main()
