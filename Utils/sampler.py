import numpy as np
from . import bezier_utils as butils
import cv2

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
    
def sample(word, results, sample_count = 15):
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