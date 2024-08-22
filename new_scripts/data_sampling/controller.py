import numpy as np
import random
from random import sample
from collections import Counter
from iphyre.simulator import IPHYRE
from iphyre.games import GAMES, PARAS
import math 
import json

from tree_search import TreeSearch

def custom_format(obj):
    if isinstance(obj, dict):
        return {k: custom_format(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return json.dumps(obj)
    else:
        return obj
    
if __name__ == '__main__':

    dict_path = "../../data/sampled_data.json"
    current_name = 1 # index
    dictionary = {}

    for idx, game in enumerate(GAMES[:10]) :
        print(f'-----  Game {idx+1}: {game} -----\n')
        demo = IPHYRE(game, fps=60) # by default, the games are from PARAS 
        tree = TreeSearch(demo, init_idx=current_name)
        tree.build_tree()
        dictionary.update(tree.output())
        current_name = tree.current_idx + 1
    
    formatted_data = custom_format(dictionary)
    with open(f'{dict_path}', 'w') as f:
        json.dump(formatted_data, f, indent=4)
            