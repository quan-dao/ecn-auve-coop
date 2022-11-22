import numpy as np
import matplotlib.pyplot as plt
from exercises.utils import *
from armen_v2x.dataset.v2x_sim.v2x_sim_utils import get_annotated_boxes_in_sensor_frame, get_available_point_clouds, \
    CLASS_COLORS
from armen_v2x.utils.visualization import show_point_cloud
from armen_v2x.utils.geometry import orthogonal_projection, get_points_in_range


def main():
    nusc, sample_token = get_a_sample()
    sample_rec = nusc.get('sample', sample_token)
    ref_sensor_name = 'LIDAR_TOP_id_0'
    points, points_src_idx = get_available_point_clouds(nusc, sample_token, ref_sensor_name,
                                                        THRESHOLD_DISTANCE_TO_LIDAR)
    # points: (N, 3[+C])
    boxes = get_annotated_boxes_in_sensor_frame(nusc, sample_rec['data'][ref_sensor_name])
    box_colors = None  # TODO
    print('showing point cloud')
    show_point_cloud(points[:, :3], boxes[:, :7], box_colors=box_colors)

    point_cloud_range = np.array([-51.2, -51.2, -25.0, 51.2, 51.2, 3.0])
    bev_resolution = 0.2
    bev_imsize = np.ceil((point_cloud_range[3: 5] - point_cloud_range[:2]) / bev_resolution).astype(int)  # (width, height)
    points = get_points_in_range(points, point_cloud_range)
    pixels, indices_pix2points = orthogonal_projection(points, point_cloud_range, bev_resolution)

    bev_occupancy = np.zeros((bev_imsize[1], bev_imsize[0]))
    # TODO: use `pixels` to assign white color (255) to occupied pixels of `bev_occupancy`

    bev_intensity = np.zeros((bev_imsize[1], bev_imsize[0]))
    unq_pixel_ids, counts = np.unique(indices_pix2points, return_counts=True)
    pixels_intensity = np.zeros(pixels.shape[0])
    np.add.at(pixels_intensity, indices_pix2points, points[:, 3])
    pixels_intensity /= counts
    bev_intensity[pixels[:, 1], pixels[:, 0]] = pixels_intensity

    fig, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    ax[0].set_title('Occupancy')
    ax[0].imshow(bev_occupancy, cmap='gray', origin='lower')
    ax[1].set_title('Intensity')
    ax[1].imshow(bev_intensity, cmap='gray', origin='lower')
    plt.show()



if __name__ == '__main__':
    main()
