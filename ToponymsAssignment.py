from Grouper.GrouperCaller import *
import networkx as nx
from tqdm import tqdm

from Utils import sampler

def _cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def group_toponyms(results, grouper, sample_count = 15, use_style_embeddings = False):
    if use_style_embeddings:
        print("Using style embeddings")
        
    G = nx.DiGraph()
    order_observations = []

    for j in tqdm(range(len(results))):
        center_entry = results[j]
        # Find the closest sample_count points to the center
        closest_points, closest_indices = sampler.sample(center_entry, results, sample_count)

        group_ids = []
        features = []
        for i, point in zip(closest_indices, closest_points):
            point_bezier_pts = np.array(np.concatenate((np.array(point['upper_bezier_pts']), np.array(point['lower_bezier_pts'][::-1]))))

            # Flatten the bezier points into a 1D array
            point_bezier_pts = point_bezier_pts.flatten()

            features.append(point_bezier_pts)

        dict_ids = grouper.get_toponym_sequence2(features, 0)

        # Remove duplicates
        included = set()
        dict_ids = [i for i in dict_ids if i not in included and not included.add(i)]

        group_ids = [closest_indices[i] for i in dict_ids if i < len(closest_indices)]

        if len(group_ids) != 0:
            embedding_j = center_entry['style_embedding'] if use_style_embeddings else None
            for i in group_ids:
                embedding_i = results[i]['style_embedding'] if use_style_embeddings else None
                if use_style_embeddings:
                    similarity = _cosine_similarity(embedding_j, embedding_i)
                    G.add_edge(j, i, weight=similarity)
                else:
                    if not G.has_edge(j, i):
                        G.add_edge(j, i, weight=1)
                    else:
                        G[j][i]['weight'] += 1

        if len(group_ids) > 1:
            order_observations.append(group_ids)

    return G, order_observations

def minimize_observation_error_sorting(observations, n):
    # Initialize pairwise preference matrix
    pairwise_matrix = [[0] * n for _ in range(n)]

    # Update pairwise preferences based on observations
    for observation in observations:
        for i in range(len(observation)):
            for j in range(i + 1, len(observation)):
                pairwise_matrix[observation[i]][observation[j]] += 1
                pairwise_matrix[observation[j]][observation[i]] -= 1

    # Construct weighted directed graph
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if pairwise_matrix[i][j] > 0:
                G.add_edge(i, j, weight=pairwise_matrix[i][j])

    # Attempt topological sorting
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # Graph has cycles, use a heuristic to break cycles
        order = list(nx.topological_sort(nx.DiGraph(G)))

    return order

def toponym_from_graph_strong_component(results, G, order_observations):
    connected_components = list(nx.strongly_connected_components(G))

    # For each connected component, create a group of results
    grouped_results = []
    for component in connected_components:
        component = [int(i) for i in component]
        observations = [ob for ob in order_observations if all([i in component for i in ob])]

        if len(observations) == 0:
            group = [results[i] for i in component]
            grouped_results.append(group)
        else:
            observations = [[component.index(i) for i in obs] for obs in observations]
            order = minimize_observation_error_sorting(observations, len(component))
            group = [results[component[i]] for i in order]
            grouped_results.append(group)

    return grouped_results

'''
def _split_community(subgraph:nx.Graph, threshold):
    if len(subgraph) <= threshold:
        return [subgraph]
    else:
        communities = nx.algorithms.community.kernighan_lin_bisection(subgraph)
        return _split_community(subgraph.subgraph(communities[0]), threshold) + _split_community(subgraph.subgraph(communities[1]), threshold)

def toponym_from_graph_community_detection(results, G: nx.DiGraph, threshold = 10):
    communities = list(nx.weakly_connected_components(G))

    # For each connected component, create a group of results
    grouped_results = []
    splited_communities = []
    for c in communities:
        subgraph = G.subgraph(c).to_undirected()
        splited_communities.extend(_split_community(subgraph, threshold))
    
    for community in splited_communities:
        group = [results[int(i)] for i in community]
        grouped_results.append(group)

    return grouped_results
'''