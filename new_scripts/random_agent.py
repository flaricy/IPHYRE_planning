'''
to play game, run 
python ../scripts/collect_play_all.py player_name
'''

import numpy as np
import random
from random import sample

from iphyre.simulator import IPHYRE
from iphyre.games import GAMES, PARAS
import math 

if __name__ == '__main__':

    #random.seed(42)
    #np.random.seed(42)
    fps = 60
    max_simu = 2000

    for game in GAMES[:] :
        demo = IPHYRE(game, fps=fps) # by default, the games are from PARAS 
        max_time = demo.max_time 
        action_space = demo.get_action_space()

        properties = demo.get_all_property() # [max_obj_num, 9]
        obj_num = len(demo.space.bodies)
        '''  property looks like 
             [lx,       ly,       rx,       ry,       radius, eli, dyn, joint, spring] for blocks
             [centre_x, centre_y, centre_x, centre_y, radius, eli, dyn, joint, spring] for balls
             the last one is the ball
        '''
        #print(game,":\n\taction space:\n", action_space, "\n\tproperties:\n", properties)
        #print(game,":", demo.space.bodies)  ## support : [Body(Body.STATIC), Body(Body.STATIC), Body(1.0, 200.0, Body.DYNAMIC)]

        all_results = []
        for simu_idx in range(max_simu) :
            demo.reset()
            actions = []
            for i in range(1, min(obj_num, len(action_space))) : # ! only 1 ball 
                # if (action_space[i][0] == 0) : 
                #     continue
                val = random.random()
                if val < 0.5 : continue
                else :
                    t = random.random() * 0.8 * max_time
                    actions.append([action_space[i][0], action_space[i][1], t])
            res, valid_step, time_count = demo.simulate(actions)
            all_results.append([res, valid_step, time_count])
            #print(simu_idx, res, time_count, actions)
        
        success_times = np.sum([1 for res, valid_step, time_count in all_results if res == 1])
        print(game, "success times:", success_times)
