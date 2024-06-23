from Grouper.GrouperCaller_v1 import *
import networkx as nx
from tqdm import tqdm

from Utils import sampler

from Utils import visualizer
from PIL import Image

def _cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def group_toponyms(results, grouper: GrouperCaller, sample_count = 15, use_style_embeddings = False, batch_size = 128):
    if use_style_embeddings:
        print("Using style embeddings")
        
    grid,_,_,_,_ = sampler.naive_grid_generator([r['center'] for r in results], ensure_at_least_k_per_grid=5, epsilon=0.1)

    center_entry_lst = [r for r in results]
    sampled_indices = []
    features_lst = []
    dict_ids_lst = []
    for j in tqdm(range(len(results))):
        center_entry = results[j]
        # Find the closest sample_count points to the center
        closest_points, closest_indices = sampler.sample(center_entry, results, sample_count, spatial_grids = grid, query_grid=grid[j], grid_search_range=2)
        #closest_points, closest_indices = sampler.sample2(center_entry, results, sample_count)
        #vis = visualizer.PolygonVisualizer()
        #vis.canvas_from_image(Image.open("Input/paris2.jpg"))
        #vis.draw(closest_points).save(f"Results/paris2/_debug.jpg")
        features = [np.array(np.concatenate((np.array(point['upper_bezier_pts']), np.array(point['lower_bezier_pts'][::-1])))).flatten() for point in closest_points]

        sampled_indices.append(closest_indices)
        features_lst.append(features)
        #dict_ids_lst.append(grouper.get_toponym_sequence2(features, 0))

    dict_ids_lst = grouper.get_toponym_sequence_batch(features_lst, batch_size=batch_size, show_progress=True)

    G = nx.DiGraph()
    order_observations = []
    for j, dict_ids, center_entry, closest_indices in zip(range(len(dict_ids_lst)), dict_ids_lst, center_entry_lst, sampled_indices):
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

    len_edges = len(G.edges)

    order = list(range(n))

    # Attempt topological sorting
    for _ in range(len_edges):
        try:
            order = list(nx.topological_sort(G))
            return order
        except nx.NetworkXUnfeasible:
            cycle = nx.find_cycle(G)
            if not cycle:
                print("Warning: No cycle found but topological sort failed.")
                break

            # Find the edge with the least weight in the cycle
            min_weight_edge = min(cycle, key=lambda edge: G.edges[edge]['weight'])
            
            # Remove the edge with the least weight
            G.remove_edge(*min_weight_edge)

    print("Warning: Topological sort failed. Using default order.")
    return order

def toponym_from_graph_strong_component(results, G, order_observations):
    connected_components = list(nx.strongly_connected_components(G))

    # For each connected component, create a group of results
    grouped_results = []
    for component in connected_components:
        component = [int(i) for i in component]
        component_set = set(component)

        # Keep observations that are in the component, drop the rest
        observations = [ob for ob in order_observations if component_set.issuperset(ob)]
        observations = [[component.index(i) for i in obs] for obs in observations]

        # Keep observations that have at least two elements in the component, drop the rest
        #observations = [ob for ob in order_observations if len(component_set.intersection(ob)) >= 2]
        #observations = [[component.index(i) for i in obs if i in component_set] for obs in observations]

        if len(observations) == 0:
            group = [results[i] for i in component]
            grouped_results.append(group)
        else:
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