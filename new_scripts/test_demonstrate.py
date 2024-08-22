'''
The player chooses an action at each frame, which is achieved by a query at each frame.
Demonstrate the whole simulation process frame by frame
'''

import numpy as np
import random
from random import sample
from collections import Counter
from iphyre.simulator import IPHYRE
from iphyre.games import GAMES, PARAS
import math 
import json

if __name__ == "__main__":

    fps = 60

    for idx, game in enumerate(GAMES[:10]) :
        print(f'-----  Game {idx+1}: {game} -----\n\n')
        demo = IPHYRE(game, fps=fps) # by default, the games are from PARAS 
        max_time = demo.max_time 
        obj_num = len(demo.space.bodies)
        
        demo.play_debug()
            