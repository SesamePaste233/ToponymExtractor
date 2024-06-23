import numpy as np
from . import bezier_utils as butils
import cv2
import math

def _anchors(_word, type = '111222333'):
    arr = [
        _word['left'],
        _word['right'],
        _word['center']
    ]
    if type == '111222333':
        return [arr[0], arr[0], arr[0], arr[1], arr[1], arr[1], arr[2], arr[2], arr[2]]
    elif type == '123123123':
        return [arr[0], arr[1], arr[2], arr[0], arr[1], arr[2], arr[0], arr[1], arr[2]]

def naive_grid_generator(points, ensure_at_least_k_per_grid = 5, epsilon = 0.1):
    '''
        points: list of tuples, spatial points
        ensure_at_least_k_per_grid: int, minimum number of points per grid
    '''
    min_x = min(points, key = lambda x: x[0])[0]
    max_x = max(points, key = lambda x: x[0])[0]
    min_y = min(points, key = lambda x: x[1])[1]
    max_y = max(points, key = lambda x: x[1])[1]
    width = max_x - min_x
    height = max_y - min_y
    w_to_h_ratio = width / height

    min_grid_size = max(width, height) / math.ceil(math.sqrt(len(points) / max(w_to_h_ratio, 1 / w_to_h_ratio) * ensure_at_least_k_per_grid))
    max_grid_size = max(width, height) 

    # Binary search for the optimal grid size
    while max_grid_size - min_grid_size > epsilon:
        mid_grid_size = (min_grid_size + max_grid_size) / 2
        grid_x = math.ceil(width / mid_grid_size)
        grid_y = math.ceil(height / mid_grid_size)
        grid_size_x = width / grid_x * 1.001
        grid_size_y = height / grid_y * 1.001
        grid = [[0 for _ in range(grid_y)] for _ in range(grid_x)]

        for point in points:
            grid[int((point[0] - min_x) / grid_size_x)][int((point[1] - min_y) / grid_size_y)] += 1

        if all([all([grid[i][j] >= ensure_at_least_k_per_grid for j in range(grid_y)]) for i in range(grid_x)]):
            max_grid_size = mid_grid_size
        else:
            min_grid_size = mid_grid_size + 1

    grid_x = math.ceil(width / min_grid_size)
    grid_y = math.ceil(height / min_grid_size)
    grid_size_x = width / grid_x * 1.001
    grid_size_y = height / grid_y * 1.001

    grid = []
    for p in points:
        p = (p[0] - min_x, p[1] - min_y)
        grid_x_index = int(p[0] / grid_size_x)
        grid_y_index = int(p[1] / grid_size_y)
        grid.append((grid_x_index, grid_y_index))

    return grid, grid_size_x, grid_size_y, grid_x, grid_y

def sample(word, results, sample_count = 15, spatial_grids = None, query_grid = (0, 0), grid_search_range = 1):
    if spatial_grids == None:
        spatial_grids = [(0, 0) for _ in range(len(results))]
        query_grid = (0, 0)
    searching_grids = set([(query_grid[0] + i, query_grid[1] + j) for i in range(-grid_search_range, grid_search_range + 1) for j in range(-grid_search_range, grid_search_range + 1)])

    bezier_pts = []
    word_upper_bezier_x, word_upper_bezier_y = [pt[0] for pt in word['upper_bezier_pts']], [pt[1] for pt in word['upper_bezier_pts']]
    word_lower_bezier_x, word_lower_bezier_y = [pt[0] for pt in word['lower_bezier_pts']], [pt[1] for pt in word['lower_bezier_pts']]
    for i in range(4):
        bezier_pts.extend([word_upper_bezier_x[i], word_upper_bezier_y[i]])
    for i in range(4):
        bezier_pts.extend([word_lower_bezier_x[i], word_lower_bezier_y[i]])
    

    word_anchors = _anchors(word, type='111222333')
    _, nabb_l, nabb_s = butils.get_bezier_bbox_params(bezier_pts)

    nabb_l_new = nabb_l * np.linalg.norm(nabb_s) / np.linalg.norm(nabb_l)

    mat_src = np.float32([np.array(word['center']),np.array(word['center']) + nabb_l,np.array(word['center']) + nabb_s])  
    mat_dst = np.float32([np.array(word['center']),np.array(word['center']) + nabb_l_new,np.array(word['center']) + nabb_s])
    T = cv2.getAffineTransform(mat_src, mat_dst)
    

    anchor_id = [(_anchors(w, type='123123123'), i) for i, w in enumerate(results) if spatial_grids[i] in searching_grids]
    anchors_to_compare, result_indices = zip(*anchor_id)

    dist_vectors = np.array(anchors_to_compare) - np.array(word_anchors)
    dist_vectors_transformed = np.einsum('ji,akj->aki', T[:,:2], dist_vectors)
    distances = np.min(np.linalg.norm(dist_vectors_transformed, axis=2), axis=1)

    closest_indices = np.argsort(distances)[:sample_count + 1]
    closest_indices = [result_indices[i] for i in closest_indices]
    dictionary = [results[i] for i in closest_indices]

    return dictionary, closest_indices

def sample2(word, results, sample_count = 15):
    bezier_pts = []
    word_upper_bezier_x, word_upper_bezier_y = [pt[0] for pt in word['upper_bezier_pts']], [pt[1] for pt in word['upper_bezier_pts']]
    word_lower_bezier_x, word_lower_bezier_y = [pt[0] for pt in word['lower_bezier_pts']], [pt[1] for pt in word['lower_bezier_pts']]
    for i in range(4):
        bezier_pts.extend([word_upper_bezier_x[i], word_upper_bezier_y[i]])
    for i in range(4):
        bezier_pts.extend([word_lower_bezier_x[i], word_lower_bezier_y[i]])
    

    word_anchors = _anchors(word, type='111222333')
    _, nabb_l, nabb_s = butils.get_bezier_bbox_params(bezier_pts)

    nabb_l_new = nabb_l * np.linalg.norm(nabb_s) / np.linalg.norm(nabb_l)

    mat_src = np.float32([np.array(word['center']),np.array(word['center']) + nabb_l,np.array(word['center']) + nabb_s])  
    mat_dst = np.float32([np.array(word['center']),np.array(word['center']) + nabb_l_new,np.array(word['center']) + nabb_s])
    T = cv2.getAffineTransform(mat_src, mat_dst)
    

    anchors_to_compare = []
    for w in results:
        anchors_to_compare.append(_anchors(w, type='123123123'))

    dist_vectors = np.array(anchors_to_compare) - np.array(word_anchors)
    dist_vectors_transformed = np.einsum('ji,akj->aki', T[:,:2], dist_vectors)
    distances = np.min(np.linalg.norm(dist_vectors_transformed, axis=2), axis=1)

    closest_indices = np.argsort(distances)[:sample_count + 1]

    dictionary = []
    for i in closest_indices:
        dictionary.append(results[i])

    return dictionary, closest_indices