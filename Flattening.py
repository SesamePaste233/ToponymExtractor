from Utils import bezier_utils as butils
from Utils import sampler
import numpy as np
from tqdm import tqdm

def _aggregate_closest_results_iter(results, sample_count = 20, evaluate_overlapping = "any", height_multiplier = 0.3):
    '''
        results: list of entries with keys 'center_bezier_pts', 'center', 'text', 'score', 'avg_height', 'left', 'right'
    '''

    center_entry = results[0]
    avg_height = center_entry['avg_height']
    multiplier = height_multiplier

    # Find the closest sample_count points to the center
    closest_points, closest_indices = sampler.sample(center_entry, results, sample_count)
    
    group = []
    group_ids = []
    ambiguous = []
    center_bezier_pts_x = [p[0] for p in center_entry['center_bezier_pts']]
    center_bezier_pts_y = [p[1] for p in center_entry['center_bezier_pts']]
    for i, point in zip(closest_indices, closest_points):
        point_bezier_pts_x = [p[0] for p in point['center_bezier_pts']]
        point_bezier_pts_y = [p[1] for p in point['center_bezier_pts']]
        bezier_dist, is_ambiguous = butils.bezier_distance(center_bezier_pts_x, center_bezier_pts_y, point_bezier_pts_x, point_bezier_pts_y, evaluate_overlapping=evaluate_overlapping, samples = 40)
        if bezier_dist < avg_height * multiplier:
            group.append(point)
            group_ids.append(i)
            if is_ambiguous:
                ambiguous.append(True)
            else:
                ambiguous.append(False)

    return group, group_ids, ambiguous

def aggregate_closest_results(results, sample_count = 20, evaluate_overlapping = "none", height_multiplier = 0.8):
    ungrouped_results = results
    grouped_results = []
    ambiguity = []

    total = len(results)

    # Sort ungrouped results by distance between 'left' and 'right', descending
    ungrouped_results.sort(key=lambda x: np.linalg.norm(np.array(x['right']) - np.array(x['left'])), reverse=True)

    with tqdm(total=total) as pbar:
        while len(ungrouped_results) > 0:
            group, group_ids, ambiguous = _aggregate_closest_results_iter(ungrouped_results, sample_count, evaluate_overlapping=evaluate_overlapping, height_multiplier=height_multiplier)
            grouped_results.append(group)
            ambiguity.append(ambiguous)
            ungrouped_results = [result for i, result in enumerate(ungrouped_results) if i not in group_ids]
            pbar.update(len(group))

    return grouped_results, ambiguity

def normalize_adhesive(_grouped_results, ambiguity, original_image):
    result = []
    for group, ab in zip(_grouped_results, ambiguity):
        if any(ab) == False:
            group.sort(key=lambda x: np.linalg.norm(np.array(x['right']) - np.array(x['left'])), reverse=True)
            result.append(group[0])
        else:
            center_bezier_pts = group[0]['center_bezier_pts']
            center_bezier_xs = [pt[0] for pt in center_bezier_pts]
            center_bezier_ys = [pt[1] for pt in center_bezier_pts]
            width = group[0]['avg_height']*0.5
            for w, a in zip(group, ab):
                if a:
                    w_bezier_xs = [pt[0] for pt in w['center_bezier_pts']]
                    w_bezier_ys = [pt[1] for pt in w['center_bezier_pts']]
                    center_bezier_xs, center_bezier_ys = butils.glue_bezier(center_bezier_xs, center_bezier_ys, w_bezier_xs, w_bezier_ys)

            center_bezier_pts = [(x, y) for x, y in zip(center_bezier_xs, center_bezier_ys)]
            
            # Optional step to get word snippet
            #snippet, transform = butils.get_center_bezier_bbox(original_image=original_image, center_bezier_pts=center_bezier_pts, width=width,scale=1.5)

            # Take the longest word in the group                    
            new_text = max([w['text'] for w in group], key=len)
            new_avg_height = np.mean([w['avg_height'] for w in group])
            new_score = np.mean([w['score'] for w in group])

            new_result = butils.make_result(center_bezier_pts, new_avg_height, new_text, new_score)

            result.append(new_result)

    return result