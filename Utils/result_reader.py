import pandas as pd
import json
import networkx as nx

import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def read_json(json_path):
    '''
        json_path: path to a json file
        returns: a list of dictionaries with the json data
    '''
    data = pd.read_json(json_path)

    # Convert the data to a list of dictionaries, where each dictionary is a row
    data = data.to_dict(orient='records')

    return data

def save_json(data, output_path):
    '''
        data: a list of dictionaries
        output_path: path to save the json file
    '''
    data = pd.DataFrame(data)
    data.to_json(output_path)
    return

def save_json_nested(data, output_path):
    '''
        data: a list of lists of dictionaries
        output_path: path to save the json file
    '''
    with open(output_path, 'w') as f:
        json.dump(data, f, cls=NpEncoder)
    return

def read_json_nested(json_path):
    '''
        json_path: path to a json file
        returns: a list of lists of dictionaries with the json data
    '''
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_toponym_graph(G, output_path):
    '''
        G: a networkx graph object
        output_path: path to save the graph
    '''
    nx.write_gexf(G, output_path)
    return

def read_toponym_graph(graph_path):
    '''
        graph_path: path to a networkx graph object
        returns: a networkx graph object
    '''
    G = nx.read_gexf(graph_path)
    return G

def extract_toponyms_from_result_groups(results):
    toponyms = []
    for group in results:
        if len(group) != 0:
            toponym = {
                'text': ' '.join([r['text'] for r in group]),
                'center': [np.mean([r['center'][0] for r in group]), np.mean([r['center'][1] for r in group])],
                'group': group
            }

        toponyms.append(toponym)

    return toponyms