import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from armen_v2x.utils.geometry import make_tf, apply_tf, rot_z
from typing import Tuple


def show_bev_img():
    pass  # TODO


def create_cube_o3d(corners: np.ndarray, color: Tuple[float] = None):
    """
    Create a box to be visualized using open3d. Convention
    forward face: 0 - 1 - 2 - 3, backward face: 4 - 5 - 6 - 7, top face: 0 - 4 - 5 - 1
    :param corners: (8, 3) - coordinate of 8 corners
    :param color: color of the cube
    :return:
    """
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # front
        [4, 5], [5, 6], [6, 7], [7, 4],  # back
        [0, 4], [1, 5], [2, 6], [3, 7],  # connecting front & back
        [0, 2], [1, 3]  # denote forward face
    ]
    if color is None:
        colors = [[1, 0, 0] for _ in range(len(lines))]  # red
    else:
        colors = [color for _ in range(len(lines))]
    cube = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    cube.colors = o3d.utility.Vector3dVector(colors)
    return cube


def box_to_corner(box: np.ndarray) -> np.ndarray:
    """
    Compute coordinate of box's corners. Convention
    forward face: 0 - 1 - 2 - 3, backward face: 4 - 5 - 6 - 7, top face: 0 - 4 - 5 - 1

    :param box: (7) - center_x, center_y, center_z, dx, dy, dz, yaw
    :return: (8, 3)
    """
    assert box.shape == (7,), f"expect (7,), get {box.shape}"
    # TODO: compute coordinate of 8 corners in the frame relative to which the `box` is expressed
    # TODO: Notie the order of 8 corners must be according to the convention stated the function doc string
    corners = np.zeros((8, 3))  # this is just a dummy value
    return corners


def show_point_cloud(points: np.ndarray, boxes: np.ndarray = None, point_colors: np.ndarray = None,
                     box_colors: np.ndarray = None):
    """
    Visualize point cloud
    :param points: (N, 3) - x, y, z
    :param boxes: (B, 7) - center_x, center_y, center_z, dx, dy, dz, yaw
    :param point_colors: (N, 3) - r, g, b
    :param box_colors: (B, 3) - r, g, b
    """
    assert points.shape[1] == 3, f"expect (N, 3), get {points.shape}"
    assert boxes.shape[1] == 7, f"expect (B, 7), get {boxes.shape}"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if point_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(point_colors)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    obj_to_display = [pcd, frame]

    if boxes is not None:
        cubes = [create_cube_o3d(box_to_corner(boxes[i]), box_colors[i] if box_colors is not None else None)
                 for i in range(boxes.shape[0])]
        obj_to_display += cubes

    o3d.visualization.draw_geometries(obj_to_display)

