import numpy as np
import matplotlib.pyplot as plt
from exercises.utils import *
from armen_v2x.dataset.v2x_sim.v2x_sim_utils import get_annotated_boxes_in_sensor_frame, get_available_point_clouds, \
    CLASS_COLORS
from armen_v2x.utils.visualization import show_point_cloud


def main():
    nusc, sample_token = get_a_sample()
    sample_rec = nusc.get('sample', sample_token)

    # get merge point cloud
    ref_sensor_name = 'LIDAR_TOP_id_1'
    merge_points, merge_points_src_idx = get_available_point_clouds(nusc, sample_token, ref_sensor_name,
                                                                    THRESHOLD_DISTANCE_TO_LIDAR)

    # TODO: assign colors to points according to their source
    merge_points_colors = None  # TODO

    boxes = get_annotated_boxes_in_sensor_frame(nusc, sample_rec['data'][ref_sensor_name])
    box_colors = None  # TODO

    print('showing point cloud from ref_sensor')
    mask_from_ref = merge_points_src_idx == int(ref_sensor_name[-1])
    show_point_cloud(merge_points[mask_from_ref, :3], boxes[:, :7], merge_points_colors[mask_from_ref], box_colors)

    print('showing merge point cloud')
    show_point_cloud(merge_points[:, :3], boxes[:, :7], merge_points_colors, box_colors)


if __name__ == '__main__':
    main()
