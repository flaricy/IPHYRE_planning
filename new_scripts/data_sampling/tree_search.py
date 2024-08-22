from copy import deepcopy

from iphyre.simulator import IPHYRE
from iphyre.games import GAMES, PARAS
import math 
import json
import random
import numpy as np

class Node():
    def __init__(self, demo: IPHYRE, depth, idx, max_simu, parent = None, action = None, terminal = False) -> None:
        '''
        idx is the name/tag of the node
        '''
        self.demo = deepcopy(demo)
        self.depth = depth
        self.idx = idx
        self.parent = parent
        # self.children = []
        # self.reward = None
        self.terminal = self.demo.check_terminate()
        self.max_simu = max_simu
        self.res = self.rullout() # success times, scene, action prob. dist.

    def restore_scene(self):
        '''
        return: {scene dict}
        '''
        return self.demo.restore_scene()

    def generate_children(self, child_idx, pos, timestep):
        next_demo = deepcopy(self.demo)
        # print(f"call from generate_children: original properties = {next_demo.get_all_property()}")
        # print(f"pos = {pos}")
        next_demo.step(pos, timestep = timestep)
        # print(f"call from generate_children: later    properties = {next_demo.get_all_property()}")
        return Node(next_demo, self.depth + 1, child_idx, self.max_simu, parent = self, action = (pos, timestep))
    
    def rullout(self, success_bound = 50):
        '''
        return (total success times, scene, probability distribution of successful actions)
        '''
        if self.terminal:
            raise Exception('Already terminated. Rullout banned.')
        
        # print(self.demo.space.bodies)
        # print(self.demo.get_all_property())
        action_space = self.demo.get_action_space_by_property()
        print(f"actions left: {action_space}")

        # if no actions left, prune 
        if len(action_space) == 0:
            self.terminal = True
            temp_demo = deepcopy(self.demo)
            res, valid_step, time_count = temp_demo.simulate([], use_current_time = True)

            if res == 1: # success
                print(f"success_times: {success_bound}, distribution: (-1: 1)")
                return (success_bound, self.restore_scene(), {-1: 1})
            else:
                print(f"fail")
                return (0, self.restore_scene(), {-1: 0})

        # if actions left, simulate

        all_results = []
        firstFrameActions = [] # store index of each action of each simulation. if val = -1, means do nothing

        is_static = self.demo.check_static()

        success_times = 0 

        simu_idx = 0
        for simu_idx in range(self.max_simu) :
            
            temp_demo = deepcopy(self.demo)
            actions = []

            if is_static:
                firstFrameIdx = int(random.random() * 0.9999 * (len(action_space))) # 0, ..., len(action_space) - 1
            else : firstFrameIdx = int(random.random() * 0.9999 * (len(action_space)+1)) - 1 # -1, 0, ..., len(action_space) - 1

            firstFrameActions.append(firstFrameIdx)

            for i in range(len(action_space)) : 
                if i == firstFrameIdx :
                    actions.append([action_space[i][1], action_space[i][2], 1/self.demo.FPS])
                else :
                    val = random.random()
                    if val < 0.5 : continue
                    else : 
                        t = random.random() * 0.7 * (self.demo.max_time - self.demo.current_time)
                        actions.append([action_space[i][1], action_space[i][2], t])

            res, valid_step, time_count = temp_demo.simulate(actions, use_current_time = True)
            if res == 1 : 
                success_times += 1
                all_results.append([res, valid_step, time_count])
                if success_times == success_bound : break 
            else : all_results.append([res, valid_step, time_count])
            

        action_count = {}
        for i in range(len(action_space)) :
            action_count[action_space[i][0]] = 0
        if not is_static: 
            action_count[-1] = 0

        for i in range(simu_idx+1) :
            if all_results[i][0] == 1 : # success
                if firstFrameActions[i] == -1 :
                    action_count[-1] += 1
                else :
                    action_count[action_space[firstFrameActions[i]][0]] += 1

        # normalize
        action_probabilities = {-1: 0}
        if success_times > 0 :
            for action, count in action_count.items():
             action_probabilities[action] = count / success_times
        if success_times > 0 :
            print(f"sucess_times: {success_times}, distribution: {action_probabilities}")
        else : 
            print(f"fail")

        # deal with already finished results 
        if simu_idx + 1 == success_times or simu_idx + 1 == self.max_simu: 
            self.terminal = True 

        return success_times, self.restore_scene(), action_probabilities
    
class TreeSearch():
    '''
    For a static scene, the STOP operation is not allowed.
    '''
    def __init__(self, demo: IPHYRE, max_depth = 6, timestep = 3/60, init_idx = 1, max_simu = 800) -> None:
        demo.reset()
        self.max_depth = max_depth
        self.timestep = timestep
        self.current_idx = init_idx # not neccessarily start from 1
        self.max_simu = max_simu
        self.node = Node(demo, 0, init_idx, max_simu=self.max_simu)
        self.node_list = [self.node] 
        self.cnt = 1 # count 
        #self.res = {} # key: node_id, value: (success_times, scene, action_probabilities)
        self.ptr = 0
        self.leaf_index = None # record the starting index of the leaf nodes
    
    def build_tree(self, max_rullout_node = 1 << 8):
        '''
        Implement a breadth-first search to expand the tree. self.node_list is the queue. The tree is always complete. 
        '''
        print("Start building tree...")
        stop_tag = False
        while self.ptr < len(self.node_list) and self.node_list[self.ptr].depth < self.max_depth:
            if self.node_list[self.ptr].terminal: 
                self.ptr += 1
                continue

            print(f'\ncurrent ptr: {self.ptr}; Time: {self.node_list[self.ptr].demo.current_time}')
            actions = self.node_list[self.ptr].demo.get_action_space_by_property()
            # print(f'actions: {actions}')

            # eliminate an object
            for _ , pos_x, pos_y in actions:
                print(f'\nRollout: {self.current_idx + 1}, current action: {pos_x}, {pos_y}, depth: {self.node_list[self.ptr].depth}, time: {self.node_list[self.ptr].demo.current_time}')
                pos = (pos_x, pos_y)
                new_node = self.node_list[self.ptr].generate_children(self.current_idx + 1, pos, self.timestep)
                suc_times, scene, distri = new_node.res
                if suc_times > 0: # only keep the node that has solutions
                    self.current_idx += 1
                    self.node_list.append(new_node)
                    self.cnt += 1
                    if self.cnt == max_rullout_node * 2:
                        self.leaf_index = max_rullout_node
                        stop_tag = True
                        break
                    if self.leaf_index is None and new_node.depth == self.max_depth:
                        self.leaf_index = len(self.node_list) - 1
            
            if stop_tag: break 

            # do nothing 
            print(f'\nRollout: {self.current_idx + 1}, current action: do nothing')
            new_node = self.node_list[self.ptr].generate_children(self.current_idx + 1, (0., 0.), self.timestep)
            suc_times, scene, distri = new_node.res
            if suc_times > 0: # only keep the node that has solutions
                    self.current_idx += 1
                    self.node_list.append(new_node)
                    self.cnt += 1
                    if self.cnt == max_rullout_node * 2:
                        self.leaf_index = max_rullout_node
                        break                    
                    if self.leaf_index is None and new_node.depth == self.max_depth:
                        self.leaf_index = len(self.node_list) - 1

            self.ptr += 1

        ## ? randomly pick min(max_rollout_node, # of all leaf nodes) in all leaf nodes, and conduct the full rollout process, that means, recursively implement generate_child and pick an action according to the distribution 

        leaf_nodes = self.node_list[self.leaf_index:]
        num_leaf_nodes = len(leaf_nodes)
        num_selected_nodes = min(max_rullout_node, num_leaf_nodes)
        selected_indices = np.random.choice(range(num_leaf_nodes), num_selected_nodes, replace=False)

        for cnt, idx in enumerate(selected_indices):
                
                print("Rollout process: {}/{}".format(cnt+1, num_selected_nodes))

                current_node = leaf_nodes[idx]
                if current_node.terminal:
                    continue
                
                terminal = False
                while not terminal:
                    action_probs = current_node.res[2]  # get action probabilities
                    actions = list(action_probs.keys()) # indices of objects
                    probabilities = list(action_probs.values())

                    selected_action = np.random.choice(actions, p=probabilities)
                    pos, timestep = None, self.timestep
                    if selected_action == -1:
                        pos = 0., 0.
                    else :
                        pos = current_node.demo.object_idx_to_pos(selected_action)
                    
                    new_node = current_node.generate_children(self.current_idx + 1, pos, timestep)
                    if new_node.res[0] > 0:
                        current_node = new_node
                        self.current_idx += 1
                        self.node_list.append(current_node)
                        terminal = current_node.terminal
                    else : terminal = True
    
    def output(self):
        res = {}
        for node in self.node_list:
            res[str(node.idx)] = {
                'scene': node.res[1],
                'action': node.res[2]
            }
        return res