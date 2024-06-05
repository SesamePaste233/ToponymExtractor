import bezier # pip install bezier
import numpy as np

from . import fitCurves

from . import bezier_curve_fitting

def bezier_from_polyline(sample_pts_x: 'list[float]', sample_pts_y: 'list[float]', max_error = 10) -> 'tuple[list[float], list[float]]':
    # Extract x and y coordinates from sample points
    sample_pts = list(zip(sample_pts_x, sample_pts_y))
    
    # Get control points
    control_points = fitCurves.fitCurve(np.array(sample_pts), max_error)[0]
    
    cpts_x = [pt[0] for pt in control_points]
    cpts_y = [pt[1] for pt in control_points]

    return cpts_x, cpts_y

def bezier_from_polyline_v2(sample_pts_x: 'list[float]', sample_pts_y: 'list[float]') -> 'tuple[list[float], list[float]]':
    control_points = bezier_curve_fitting.get_bezier_parameters(sample_pts_x, sample_pts_y)
    return [pt[0] for pt in control_points], [pt[1] for pt in control_points]

def bezier_to_polyline(control_points_x: 'list[float]', control_points_y: 'list[float]', num_pts: int=8) -> 'tuple[list[float], list[float]]':
    nodes = np.asfortranarray([control_points_x, control_points_y])
    curve = bezier.Curve(nodes, degree=3)
    x, y = curve.evaluate_multi(np.linspace(0, 1, num_pts))
    return list(x), list(y)

def bezier_length(control_points_x: 'list[float]', control_points_y: 'list[float]') -> float:
    nodes = np.asfortranarray([control_points_x, control_points_y])
    curve = bezier.Curve(nodes, degree=3)
    return curve.length

def closest_pts_ids(xx1, yy1, xx2, yy2):
    from scipy.spatial.distance import cdist    
    n_pts = len(xx1)

    array1 = np.array([xx1, yy1]).T
    array2 = np.array([xx2, yy2]).T

    points = np.vstack((array1, array2))

    p_dist = cdist(points, points, 'euclidean')

    distances = p_dist[:n_pts, n_pts:]

    agm1 = np.argmin(distances, axis = 1)

    agm2 = np.argmin(distances, axis = 0)

    return agm1, agm2

def calc_overlapping_segment(agm, extend_endpoints = False):
    curve_1_markers_end = (agm == 0) | (agm == len(agm) - 1)

    # Get the maximum continuous False sequence from curve_1_markers_end1
    max_false_seq = []
    current_false_seq = []
    for i, marker in enumerate(curve_1_markers_end):
        if marker == False:
            current_false_seq += [i]
        else:
            if len(current_false_seq) > len(max_false_seq):
                max_false_seq = current_false_seq
            current_false_seq = []
    if len(current_false_seq) > len(max_false_seq):
        max_false_seq = current_false_seq

    if extend_endpoints:
        if max_false_seq[0] == 1:
            max_false_seq = [0] + max_false_seq
        elif max_false_seq[-1] == len(agm) - 2:
            max_false_seq += [len(agm) - 1]

    return max_false_seq

def calc_overlapping_endpoint(poly_x1, poly_y1, poly_x2, poly_y2):
    curve_1_markers_end1 = []
    curve_1_markers_end2 = []
    for x1, y1 in zip(poly_x1, poly_y1):
        dists = [np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) for x2, y2 in zip(poly_x2, poly_y2)]
        min_pos = np.argmin(dists)
        if min_pos == 0:
            curve_1_markers_end1.append(False)
            curve_1_markers_end2.append(True)
        elif min_pos == len(poly_x2) - 1:
            curve_1_markers_end1.append(True)
            curve_1_markers_end2.append(False)
        else:
            curve_1_markers_end1.append(False)
            curve_1_markers_end2.append(False)

    # Get the maximum continuous False sequence from curve_1_markers_end1
    max_false_seq = []
    current_false_seq = []
    for i, marker in enumerate(curve_1_markers_end1):
        if marker == False:
            current_false_seq += [i]
        else:
            if len(current_false_seq) > len(max_false_seq):
                max_false_seq = current_false_seq
            current_false_seq = []
    if len(current_false_seq) > len(max_false_seq):
        max_false_seq = current_false_seq

    current_false_seq = []
    for i, marker in enumerate(curve_1_markers_end2):
        if marker == False:
            current_false_seq += [i]
        else:
            if len(current_false_seq) > len(max_false_seq):
                max_false_seq = current_false_seq
            current_false_seq = []
    if len(current_false_seq) > len(max_false_seq):
        max_false_seq = current_false_seq

    return max_false_seq

def bezier_distance(control_points_x1: 'list[float]', control_points_y1: 'list[float]', control_points_x2: 'list[float]', control_points_y2: 'list[float]', evaluate_overlapping = "none", samples = 32, _ambiguous_mark = False, ambiguity_threshold = 0.1) -> 'tuple[float, bool]':
    '''
        Compute Frechet distance between two bezier curves
        evaluate_overlapping: "none" - no overlapping, "any" - any overlapping segment, neglecting the rest, "endpoint" - evaluate only the segment containing endpoints
    '''
    import frechetdist as fd
    samples = samples

    poly_x1, poly_y1 = bezier_to_polyline(control_points_x1, control_points_y1, num_pts = samples)
    poly_x2, poly_y2 = bezier_to_polyline(control_points_x2, control_points_y2, num_pts = samples)

    if evaluate_overlapping == "endpoint":
        # From every point in 1, find the closest point in 2
        overlapping_1to2 = calc_overlapping_endpoint(poly_x1, poly_y1, poly_x2, poly_y2)
        overlapping_2to1 = calc_overlapping_endpoint(poly_x2, poly_y2, poly_x1, poly_y1)
        poly_x1 = [poly_x1[i] for i in range(len(poly_x1)) if i in overlapping_1to2]
        poly_y1 = [poly_y1[i] for i in range(len(poly_y1)) if i in overlapping_1to2]
        poly_x2 = [poly_x2[i] for i in range(len(poly_x2)) if i in overlapping_2to1]
        poly_y2 = [poly_y2[i] for i in range(len(poly_y2)) if i in overlapping_2to1]
    elif evaluate_overlapping == "any":
        agm1, agm2 = closest_pts_ids(poly_x1, poly_y1, poly_x2, poly_y2)
        overlapping_1to2 = calc_overlapping_segment(agm1)
        overlapping_2to1 = calc_overlapping_segment(agm2)
        poly_x1 = [poly_x1[i] for i in range(len(poly_x1)) if i in overlapping_1to2]
        poly_y1 = [poly_y1[i] for i in range(len(poly_y1)) if i in overlapping_1to2]
        poly_x2 = [poly_x2[i] for i in range(len(poly_x2)) if i in overlapping_2to1]
        poly_y2 = [poly_y2[i] for i in range(len(poly_y2)) if i in overlapping_2to1]
    
    if evaluate_overlapping != "none":
        thresh = 3
        ambiguous_mark = False
        if len(poly_x1) < thresh or len(poly_x2) < thresh:
            return np.inf, _ambiguous_mark

        bezier_1_x, bezier_1_y = bezier_from_polyline_v2(poly_x1, poly_y1)
        bezier_2_x, bezier_2_y = bezier_from_polyline_v2(poly_x2, poly_y2)

        blen_1 = bezier_length(control_points_x1, control_points_y1)
        blen_2 = bezier_length(control_points_x2, control_points_y2)
        blen_ol = bezier_length(bezier_1_x, bezier_1_y)

        if blen_2 - blen_ol > ambiguity_threshold * (blen_2 + blen_1 - blen_ol):
            ambiguous_mark = True

        return bezier_distance(bezier_1_x, bezier_1_y, bezier_2_x, bezier_2_y, evaluate_overlapping="none", _ambiguous_mark = ambiguous_mark)

    P = np.array([poly_x1, poly_y1]).T
    Q = np.array([poly_x2, poly_y2]).T

    dist = fd.frdist(P, Q)

    return dist, _ambiguous_mark


def polyline_length(sample_pts_x: 'list[float]', sample_pts_y: 'list[float]') -> float:
    length = 0
    for i in range(1, len(sample_pts_x)):
        length += np.linalg.norm(np.array([sample_pts_x[i], sample_pts_y[i]]) - np.array([sample_pts_x[i-1], sample_pts_y[i-1]]))
    return length

def _get_bbox_vertices(pts, angle):
    mean = np.float32([pts[:, 0].mean(), pts[:, 1].mean()])
    c, s = np.cos(angle), np.sin(angle)
    R = np.float32([c, -s, s, c]).reshape(2, 2)
    pts = (pts.astype(np.float32) - mean) @ R
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    corners = np.float32([x0, y0, x0, y1, x1, y1, x1, y0])
    corners = corners.reshape(-1, 2) @ R.T + mean
    return corners

from PIL import Image
import cv2

def get_bezier_bbox_params(bezier_pts):
    upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
    lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

    upper_half_x = [p[0] for p in upper_half]
    upper_half_y = [p[1] for p in upper_half]
    lower_half_x = [p[0] for p in lower_half]
    lower_half_y = [p[1] for p in lower_half]

    poly_upper_x, poly_upper_y = bezier_to_polyline(upper_half_x, upper_half_y)
    poly_lower_x, poly_lower_y = bezier_to_polyline(lower_half_x, lower_half_y)

    xs, ys = poly_upper_x + poly_lower_x[::-1], poly_upper_y + poly_lower_y[::-1]

    # Generate non-axis aligned bounding box
    angle = np.arctan2(upper_half_y[0] - upper_half_y[-1], upper_half_x[0] - upper_half_x[-1])

    # Get bbox vertices
    corners_nabb = _get_bbox_vertices(np.array([xs, ys]).T, angle)

    vector_long = corners_nabb[0] - corners_nabb[1]
    vector_short = corners_nabb[1] - corners_nabb[2]

    return corners_nabb, vector_long, vector_short

def greedy_sort(pts, start_id = 0):
    # Find the closest point to the start_id
    sorted_pts = [pts[start_id]]
    pts = np.delete(pts, start_id, axis = 0)
    while len(pts) > 0:
        dists = np.linalg.norm(pts - sorted_pts[-1], axis = 1)
        min_pos = np.argmin(dists)
        sorted_pts.append(pts[min_pos])
        pts = np.delete(pts, min_pos, axis = 0)
    return sorted_pts

def glue_bezier(control_points_x1: 'list[float]', control_points_y1: 'list[float]', control_points_x2: 'list[float]', control_points_y2: 'list[float]', samples = 10):
    poly_x1, poly_y1 = bezier_to_polyline(control_points_x1, control_points_y1, num_pts = samples)
    poly_x2, poly_y2 = bezier_to_polyline(control_points_x2, control_points_y2, num_pts = samples)
    
    dir_vec = np.array([poly_x1[-1] - poly_x1[0], poly_y1[-1] - poly_y1[0]])
    pts = np.array([poly_x1 + poly_x2, poly_y1 + poly_y2]).T
    center = np.mean(pts, axis = 0)

    min_pos = np.argmin(np.dot(pts - center, dir_vec))
    
    sorted_pts = greedy_sort(pts, min_pos)

    poly_x = [pt[0] for pt in sorted_pts]
    poly_y = [pt[1] for pt in sorted_pts]

    return bezier_from_polyline_v2(poly_x, poly_y)

def get_multi_bezier_bbox(original_image: Image.Image, bezier_pts_lst, width, scale = 1):
    xs, ys = [], []

    angle = 0
    weight = 0

    for i, bezier_pts in enumerate(bezier_pts_lst):
        poly_x, poly_y = bezier_to_polyline([p[0] for p in bezier_pts], [p[1] for p in bezier_pts])
        length = polyline_length(poly_x, poly_y)
        _angle = np.arctan2(poly_y[0] - poly_y[-1], poly_x[0] - poly_x[-1])
        if i == 0:
            angle = _angle

        offset_vec = np.array([np.cos(_angle + np.pi/2), np.sin(_angle + np.pi/2)]) * width
        poly_center_x_upper = [x + offset_vec[0] for x in poly_x]
        poly_center_y_upper = [y + offset_vec[1] for y in poly_y]
        poly_center_x_lower = [x - offset_vec[0] for x in poly_x]
        poly_center_y_lower = [y - offset_vec[1] for y in poly_y]
        combined_x = poly_center_x_upper + poly_center_x_lower[::-1]
        combined_y = poly_center_y_upper + poly_center_y_lower[::-1]
        xs.extend(combined_x)
        ys.extend(combined_y)


    center = [np.mean(xs), np.mean(ys)]
    xs = [scale * (x - center[0]) + center[0] for x in xs]
    ys = [scale * (y - center[1]) + center[1] for y in ys]

    return _get_bbox_impl(original_image, xs, ys, angle)

def make_result(center_bezier_pts, width, text, score):
    center_bezier_xs = [p[0] for p in center_bezier_pts]
    center_bezier_ys = [p[1] for p in center_bezier_pts]

    angle = np.arctan2(center_bezier_ys[0] - center_bezier_ys[-1], center_bezier_xs[0] - center_bezier_xs[-1])

    # Offset vector, perpendicular to the angle and length of width
    offset_vec = np.array([np.cos(angle + np.pi/2), np.sin(angle + np.pi/2)]) * width * 0.5
    
    poly_center_x, poly_center_y = bezier_to_polyline(center_bezier_xs, center_bezier_ys, num_pts = 25)
    poly_center_x_upper = [x + offset_vec[0] for x in poly_center_x]
    poly_center_y_upper = [y + offset_vec[1] for y in poly_center_y]
    poly_center_x_lower = [x - offset_vec[0] for x in poly_center_x][::-1]
    poly_center_y_lower = [y - offset_vec[1] for y in poly_center_y][::-1]
    polygon_x = poly_center_x_upper + poly_center_x_lower
    polygon_y = poly_center_y_upper + poly_center_y_lower

    upper_bezier_x, upper_bezier_y = bezier_from_polyline_v2(poly_center_x_upper, poly_center_y_upper)
    lower_bezier_x, lower_bezier_y = bezier_from_polyline_v2(poly_center_x_lower, poly_center_y_lower)

    center = ((center_bezier_xs[0] + center_bezier_xs[-1])/2, (center_bezier_ys[0] + center_bezier_ys[-1])/2)
    left = (center_bezier_xs[0], center_bezier_ys[0])
    right = (center_bezier_xs[-1], center_bezier_ys[-1])

    new_result = {
        "polygon_x": polygon_x,
        "polygon_y": polygon_y,
        "upper_bezier_pts": list(zip(upper_bezier_x, upper_bezier_y)),
        "center_bezier_pts": center_bezier_pts,
        "lower_bezier_pts": list(zip(lower_bezier_x, lower_bezier_y)),
        "text": text,
        "score": score,
        "avg_height": width,
        "center": center,
        "left": left,
        "right": right
    }

    return new_result

def get_center_bezier_bbox(original_image: Image.Image, center_bezier_pts, width, scale = 1):
    center_x = [p[0] for p in center_bezier_pts]
    center_y = [p[1] for p in center_bezier_pts]
    angle = np.arctan2(center_y[0] - center_y[-1], center_x[0] - center_x[-1])

    # Offset vector, perpendicular to the angle and length of width
    offset_vec = np.array([np.cos(angle + np.pi/2), np.sin(angle + np.pi/2)]) * width
    
    poly_center_x, poly_center_y = bezier_to_polyline(center_x, center_y)
    poly_center_x_upper = [x + offset_vec[0] for x in poly_center_x]
    poly_center_y_upper = [y + offset_vec[1] for y in poly_center_y]
    poly_center_x_lower = [x - offset_vec[0] for x in poly_center_x]
    poly_center_y_lower = [y - offset_vec[1] for y in poly_center_y]
    combined_x = poly_center_x_upper + poly_center_x_lower[::-1]
    combined_y = poly_center_y_upper + poly_center_y_lower[::-1]

    return _get_bbox_impl(original_image, combined_x, combined_y, angle)

def get_bezier_bbox(original_image: Image.Image, upper_half: 'list[tuple[float]]', lower_half: 'list[tuple[float]]', reverse_lower = True, scale = 1):
    upper_half_x = [p[0] for p in upper_half]
    upper_half_y = [p[1] for p in upper_half]
    lower_half_x = [p[0] for p in lower_half]
    lower_half_y = [p[1] for p in lower_half]

    center = [np.mean(upper_half_x + lower_half_x), np.mean(upper_half_y + lower_half_y)]

    upper_half_x = [scale * (p[0] - center[0]) + center[0] for p in upper_half]
    upper_half_y = [scale * (p[1] - center[1]) + center[1] for p in upper_half]
    lower_half_x = [scale * (p[0] - center[0]) + center[0] for p in lower_half]
    lower_half_y = [scale * (p[1] - center[1]) + center[1] for p in lower_half]

    poly_upper_x, poly_upper_y = bezier_to_polyline(upper_half_x, upper_half_y)
    poly_lower_x, poly_lower_y = bezier_to_polyline(lower_half_x, lower_half_y)

    if reverse_lower:
        xs, ys = poly_upper_x + poly_lower_x[::-1], poly_upper_y + poly_lower_y[::-1]
    else:
        xs, ys = poly_upper_x + poly_lower_x, poly_upper_y + poly_lower_y

    # Generate non-axis aligned bounding box
    udy, udx = upper_half_y[0] - upper_half_y[-1], upper_half_x[0] - upper_half_x[-1]
    angle = np.arctan2(udy, udx)

    return _get_bbox_impl(original_image, xs, ys, angle)

def _get_bbox_impl(original_image: Image.Image, xs, ys, angle):
    '''
    xs, ys: list[float] vertices of convex shape, clockwise order
    '''
    # Plot xs, ys, angle on the image, angle as a line from the center of the shape
    #import matplotlib.pyplot as plt
    #plt.imshow(original_image)
    #plt.plot(xs, ys, 'bo')
    #plt.plot([np.mean(xs)], [np.mean(ys)], 'ro')
    #plt.plot([np.mean(xs), np.mean(xs) + 100 * np.cos(angle)], [np.mean(ys), np.mean(ys) + 100 * np.sin(angle)], 'r')
    #plt.show()

    # Get bbox vertices
    corners_nabb = _get_bbox_vertices(np.array([xs, ys]).T, angle)

    corners_aabb = np.array([corners_nabb.min(axis=0), corners_nabb.max(axis=0)])
    
    center = (corners_aabb[0] + corners_aabb[1]) / 2
    length = np.linalg.norm(corners_aabb[0] - corners_aabb[1])
    corners_aabb[0] = center - length * 1.0 / 2
    corners_aabb[1] = center + length * 1.0 / 2

    # Crop original image
    corners_aabb = corners_aabb.astype(int)
    cropped_image = original_image.crop((corners_aabb[0][0], corners_aabb[0][1], corners_aabb[1][0], corners_aabb[1][1]))

    #plt.imshow(cropped_image)
    #plt.show()

    # Convert nabb to aabb's coordinate system
    corners_nabb -= corners_aabb[0]
    T0 = np.float32([[1, 0, -corners_aabb[0][0]], [0, 1, -corners_aabb[0][1]], [0, 0, 1]])

    # Rotate cropped image and the nabb
    cropped_image = np.array(cropped_image)
    M = cv2.getRotationMatrix2D((cropped_image.shape[1]//2, cropped_image.shape[0]//2), np.degrees(angle) + 180, 1)
    rotated_image = cv2.warpAffine(cropped_image, M, (cropped_image.shape[1], cropped_image.shape[0]))

    corners_nabb = np.hstack([corners_nabb, np.ones((4, 1))]).T

    corners_nabb = M @ corners_nabb

    #plt.imshow(rotated_image)
    #plt.show()

    # Crop the rotated image    
    corners_nabb = corners_nabb.astype(int)

    # Clamp the corners
    corners_nabb[0] = np.clip(corners_nabb[0], 0, rotated_image.shape[1])
    corners_nabb[1] = np.clip(corners_nabb[1], 0, rotated_image.shape[0])

    cropped_rotated_image = rotated_image[corners_nabb[1].min():corners_nabb[1].max(), corners_nabb[0].min():corners_nabb[0].max()]

    # Translation matrix for cropping
    T = np.float32([[1, 0, -corners_nabb[0].min()], [0, 1, -corners_nabb[1].min()], [0, 0, 1]])

    # Combine rotation and translation
    M = T @ np.concatenate((M, np.array([[0, 0, 1]])), axis=0) @ T0

    # Return the image as PIL Image
    return Image.fromarray(cropped_rotated_image), M

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = Image.open('test.png')
    center_bezier = [[0, 0], [30, 30], [60, 60], [100, 100]]

    snippet, t = get_center_bezier_bbox(image, center_bezier, 10)

    plt.imshow(snippet)
    plt.show()