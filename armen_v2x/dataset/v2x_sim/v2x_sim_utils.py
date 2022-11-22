import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from einops import rearrange
from armen_v2x.utils.geometry import make_tf, quaternion_yaw, apply_tf


CLASS_NAMES = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
CLASS_COLORS = plt.cm.rainbow(np.linspace(0, 1, len(CLASS_NAMES)))[:, :3]
CLASS_NAME_TO_COLOR = dict(zip(CLASS_NAMES, CLASS_COLORS))
CLASS_NAME_TO_INDEX = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


def get_point_cloud(nusc: NuScenes, lidar_token: str, thresh_dist_to_lidar: float) -> np.ndarray:
    """
    Get a NuScenes point cloud. Note point cloud is expressed in LiDAR's frame
    :param nusc: NuScenes API
    :param lidar_token:
    :param thresh_dist_to_lidar: in meter, to remove points too close to LiDAR according to
        their distance to LIDAR on XY plane
    :return:
        - point_cloud: (N, 4) - x, y, z, intensity
    """
    point_cloud_file = nusc.get_sample_data_path(lidar_token)
    points = np.fromfile(point_cloud_file, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]  # (x, y, z, intensity)

    mask_valid = LA.norm(points[:, :2], axis=1) > thresh_dist_to_lidar
    return points[mask_valid]


def get_tf_vehicle_from_sensor(nusc: NuScenes, sensor_token: str) -> np.ndarray:
    """
    Get transformation that map points in sensor frame to ego vehicle frame
    :param nusc: NuScenes API
    :param sensor_token: sample data token
    :return:
        - vehicle_from_sensor: (4, 4)
    """
    sensor_record = nusc.get('sample_data', sensor_token)
    calib_record = nusc.get('calibrated_sensor', sensor_record['calibrated_sensor_token'])
    vehicle_from_sensor = make_tf(calib_record['translation'], calib_record['rotation'])
    return vehicle_from_sensor


def get_tf_global_from_sensor(nusc: NuScenes, sensor_token: str) -> np.ndarray:
    """
    Get transformation that map points in sensor frame to global frame
    :param nusc: NuScenes API
    :param sensor_token: sample data token
    :return:
        - glob_from_sensor: (4, 4)
    """
    vehicle_from_sensor = get_tf_vehicle_from_sensor(nusc, sensor_token)
    sensor_record = nusc.get('sample_data', sensor_token)
    vehicle_record = nusc.get('ego_pose', sensor_record['ego_pose_token'])
    glob_from_vehicle = make_tf(vehicle_record['translation'], vehicle_record['rotation'])
    glob_from_sensor = glob_from_vehicle @ vehicle_from_sensor
    return glob_from_sensor


def get_annotated_boxes(nusc: NuScenes, sensor_token: str, ignored_names: list = None) -> np.ndarray:
    """
    Get annotated boxes @ timestamp of sensor. Note: annotated boxes are expressed in GLOBAL frame
    :param nusc: NuScenes API
    :param sensor_token: sample data token
    :param ignored_names: classes that are ignored
    :return:
        - boxes: (N, 8) - center_x, center_y, center_z, dx, dy, dz, yaw, class_name
    """
    if ignored_names is None:
        ignored_names = ['ignore']
    if 'ignore' not in ignored_names:
        ignored_names.append('ignore')
    boxes = []
    annos = nusc.get_boxes(sensor_token)
    for anno in annos:
        det_name = map_name_from_general_to_detection[anno.name]
        if det_name in ignored_names:
            continue

        # remove spurious anno (i.e. zero volume)
        if anno.wlh[0] * anno.wlh[1] * anno.wlh[2] < 1e-1:
            continue

        # remove empty box
        anno_rec = nusc.get('sample_annotation', anno.token)
        if max(anno_rec['num_lidar_pts']) < 1:
            continue

        cur_box = [*anno.center.tolist(), anno.wlh[1], anno.wlh[0], anno.wlh[2], quaternion_yaw(anno.orientation),
                   CLASS_NAME_TO_INDEX[det_name]]
        boxes.append(cur_box)

    return np.array(boxes).astype(float)


def get_annotated_boxes_in_sensor_frame(nusc: NuScenes, sensor_token: str, ignored_names: list = None) -> np.ndarray:
    """
    Get annotated boxes @ timestamp of sensor, in SENSOR frame  TODO
    :param nusc: NuScenes API
    :param sensor_token: sample data token
    :param ignored_names: classes that are ignored
    :return:
        - boxes: (N, 8) - center_x, center_y, center_z, dx, dy, dz, yaw, class_name
    """
    boxes = get_annotated_boxes(nusc, sensor_token, ignored_names)  # (N, 8) - in global frame

    # TODO: map `boxes` from global frame to sensor frame (sensor is represented by `sensor_token`)
    return boxes


def get_available_lidar_tokens(nusc: NuScenes, sample_token: str) -> dict:
    """
    Get tokens of LiDAR available @ the inputted sample
    :param nusc: NuScenes API
    :param sample_token:
    :return:
        - {channel: token}
    """
    sample_rec = nusc.get('sample', sample_token)
    out = dict()
    for channel, token in sample_rec['data'].items():
        if 'LIDAR_TOP' in channel and 'SEM' not in channel:
            out[channel] = token
    return out


def get_available_point_clouds(nusc: NuScenes, sample_token: str, ref_sensor_name: str, thresh_dist_to_lidar: float) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Get point clouds available @ the inputted sample & map them to the frame of ref_sensor_name
    :param nusc: NuScenes API
    :param sample_token:
    :param ref_sensor_name: name of the LiDAR that is chosen to be reference frame (e.g., LIDAR_TOP_id_1)
    :param thresh_dist_to_lidar: distance threshold to remove points too close to LiDAR
    :return:
        - merge_points (N_tot, 3[+C]) - x, y, z, C-dim feat
        - merge_points_src_idx (N_tot,) - points_src_idx[i] = j means merge_points[i] is collected by LIDAR_TOP_id_{j}
    """
    lidar_names2tokens = get_available_lidar_tokens(nusc, sample_token)
    ref_sensor_token = lidar_names2tokens[ref_sensor_name]
    # TODO: get point clouds from every lidar stored in lidar_names2tokens & map them to ref_sensor
    # TODO: point clouds, after being mapped to ref_sensor frame, are stored in `points_list`
    # TODO: `points_src_idx` stores index of the LiDAR that collects each point in each point clouds. This means
    # TODO: if the first point cloud has 3 points and is collected by LiDAR "1", points_src_idx[0] = np.array([1, 1, 1])
    points_list, points_src_idx = [], []

    merge_points = np.concatenate(points_list)
    merge_points_src_idx = np.concatenate(points_src_idx)
    return merge_points, merge_points_src_idx
