import matplotlib.pyplot as plt
import PIL
from exercises.utils import *
from armen_v2x.dataset.v2x_sim.v2x_sim_utils import get_point_cloud, get_annotated_boxes_in_sensor_frame, \
    CLASS_COLORS
from armen_v2x.utils.visualization import show_point_cloud


def main():
    nusc, sample_token = get_a_sample()
    sample_rec = nusc.get('sample', sample_token)
    print_record(sample_rec)

    print('showing IRSU images')
    fig, ax = plt.subplots(3, 3)
    cam_idx_to_ax = [(1, 2), (2, 1), (1, 0), (0, 1)]
    for r_idx in range(3):
        for c_idx in range(3):
            ax[r_idx, c_idx].set_xticks([])
            ax[r_idx, c_idx].set_yticks([])
    for cam_idx in range(4):
        img = PIL.Image.open(nusc.get_sample_data_path(sample_rec['data'][f'CAM_id_0_{cam_idx}']))
        _r, _c = cam_idx_to_ax[cam_idx]
        ax[_r, _c].imshow(img)
        ax[_r, _c].set_title(f'CAM_id_0_{cam_idx}')

    plt.show()

    thresh_dist = THRESHOLD_DISTANCE_TO_LIDAR
    print('showing IRSU point cloud')
    irsu_token = sample_rec['data']['LIDAR_TOP_id_0']
    irsu_points = get_point_cloud(nusc, irsu_token, thresh_dist)
    irsu_boxes = get_annotated_boxes_in_sensor_frame(nusc, irsu_token)
    box_colors = None  # TODO
    show_point_cloud(irsu_points[:, :3], irsu_boxes[:, :7], box_colors=box_colors)


if __name__ == '__main__':
    main()
