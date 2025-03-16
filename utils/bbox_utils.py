def get_centre_of_bbox(bbox):
    """
    Get the center coordinates of a bounding box."
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def measure_distance(p1, p2):
    """
    Measure the distance between two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_foot_position(bbox):
    """
    Get the foot position of a bounding box.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    """
    Get the index of the closest keypoint to a given point.
    """
    closest_distance = float('inf')
    keypoint_ind = keypoint_indices[0]

    for index in keypoint_indices:
        keypoint = keypoints[index*2], keypoints[index*2 + 1]
        distances = abs(point[1] - keypoint[1])

        if distances < closest_distance:
            closest_distance = distances
            keypoint_ind = index

    return keypoint_ind


def get_height_of_bbox(bbox):
    """
    Get the height of a bounding box.
    """
    return bbox[3] - bbox[1]


def measure_xy_distance(p1, p2):
    """
    Measure the distance between two points in pixels.
    """
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
