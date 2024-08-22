'''
    In the first frame, the agent must select an action or do nothing with equal probability. In later frames, the agent follows the policy of random_agent.py .
'''

import numpy as np
import random
from random import sample
from collections import Counter
from iphyre.simulator import IPHYRE
from iphyre.games import GAMES, PARAS
import math 
import json

if __name__ == '__main__':

    dict_path = '../data/firstFrameDistribution.json'

    #random.seed(42)
    #np.random.seed(42)
    fps = 60
    max_simu = 5000

    successAction_dict = {} # count the number of each action in each game on the first frame for all successful simulations 

    for game in GAMES[:10] :
        demo = IPHYRE(game, fps=fps) # by default, the games are from PARAS 
        max_time = demo.max_time 
        obj_num = len(demo.space.bodies)

        action_space = demo.get_action_space_by_property()
        '''
            action_space looks like [(idx, pos_x, pos_y)]
        '''
        
        properties = demo.get_all_property() # [max_obj_num, 9]
        
        '''  property looks like 
             [lx,       ly,       rx,       ry,       radius, eli, dyn, joint, spring] for blocks
             [centre_x, centre_y, centre_x, centre_y, radius, eli, dyn, joint, spring] for balls
             the last one is the ball
        '''
        print(game,":\naction space:\n", action_space, "\nproperties:\n", properties)
        #print(game,":", demo.space.bodies)  ## support : [Body(Body.STATIC), Body(Body.STATIC), Body(1.0, 200.0, Body.DYNAMIC)]

        
        all_results = []
        firstFrameActions = [] # store index of each action of each simulation. if val = -1, means do nothing

        for simu_idx in range(max_simu) :

            demo.reset()
            actions = []

            firstFrameIdx = int(random.random() * 0.9999 * (len(action_space)+1)) - 1 # -1, 0, ..., len(action_space) - 1
            firstFrameActions.append(firstFrameIdx)

            for i in range(len(action_space)) : 
                if i == firstFrameIdx :
                    actions.append([action_space[i][1], action_space[i][2], 1/fps])
                else :
                    val = random.random()
                    if val < 0.5 : continue
                    else : 
                        t = random.random() * 0.7 * max_time
                        actions.append([action_space[i][1], action_space[i][2], t])

            res, valid_step, time_count = demo.simulate(actions)
            all_results.append([res, valid_step, time_count])
            #print(simu_idx, res, time_count, actions)
        
        # 统计成功模拟中 firstFrameActions 的情况
        action_count = {-1: 0}
        for i in range(len(action_space)) :
            action_count[action_space[i][0]] = 0

        success_times = 0
        for i in range(max_simu) :
            if all_results[i][0] == 1 : # success
                success_times += 1
                if firstFrameActions[i] == -1 :
                    action_count[-1] += 1
                else :
                    action_count[action_space[firstFrameActions[i]][0]] += 1
        print(action_count)

        # normalize
        action_probabilities = {}
        for action, count in action_count.items():
             action_probabilities[action] = count / success_times
        
        successAction_dict[game] = action_probabilities

        print(game, "success times:", success_times, "\npolicy distribution:", action_probabilities)
        print("\n\n")

    with open(dict_path, "w") as f:
        json.dump(successAction_dict, f, indent=4)