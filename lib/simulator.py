import sys
import time
import pygame
from pygame.locals import *
import pymunk
import pymunk.pygame_util
import numpy as np
import os
from copy import deepcopy
import json

from iphyre.games import PARAS, MAX_OBJ_NUM, MAX_ELI_OBJ_NUM


class IPHYRE():
    def __init__(self, game='support', fps=60, src=PARAS):
        # general information
        self.game = game
        self.HEIGHT, self.WIDTH = 600, 600
        self.FPS = fps
        self.timestep = 1 / self.FPS 
        self.current_time = 0
        self.max_time = 15
        self.max_obj_num = MAX_OBJ_NUM
        self.max_eli_obj_num = MAX_ELI_OBJ_NUM
        self.PARAS = src

        self.b_mass, self.b_elasticity, self.b_friction = 1.0, 0.1, 0.5 # ball properties
        self.l_friction, self.l_elasticity = 0.5, 0.1 # line properties
        self.space = pymunk.Space()
        self.space.gravity = (0., 100.0)
        self.screen = None

        # game reward, it encourages using less time and eliminating less blocks
        self.step_reward = -1 / self.FPS
        self.eli_reward = -10
        self.success_reward = 1000

        # specific information of games
        self.blocks = self.PARAS[self.game]['block']  # [[[x1, y1], [x2, y2]],[...]]
        self.balls = self.PARAS[self.game]['ball'] # [[x, y, radius]]
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls) # [1, 1, ..., 1, 0, 0, ..., 0]
        self.num_ball = len(self.balls)
        self.eli = deepcopy(self.PARAS[self.game]['eli'])
        self.eli_mask = None
        self.dynamic = deepcopy(self.PARAS[self.game]['dynamic'])
        self.num_obj = len(self.eli)

        self.joint = None
        if 'joint' in self.PARAS[self.game].keys():
            self.joint = self.PARAS[self.game]['joint']
        self.spring = None
        if 'spring' in self.PARAS[self.game].keys():
            self.spring = self.PARAS[self.game]['spring']
       
    def check_terminate(self):
        if self.examine_success() or self.current_time > self.max_time:
            return True
        return False
    
    def add_constraint_to_body(self, type, index, line, cnt, dic):
        if type == 'joint':
                pos = -1
        elif type == 'spring':
            pos = -2
        else: 
            print("Invalid type name")
            return False
        if index >= 0 and index < line:
            lst = 'block'
            idx = index
        else:
            lst = 'ball'
            idx = index - line
        dic[lst][idx][pos] = cnt
        return True
            
    def restore_scene(self, append_index = False):
        '''
        Return the current state in the form of a dict, looking like:
        'support':
            {'block': [[200., 400., 300., 400., eli, dyn, vx, vy, w_rotate, spring, joint], ...],
            'ball': [[250., 340., 20., vx, vy, w_rotate, spring, joint], ...]
            }
        '''
        state = {}
        state[self.game] = {}
        state[self.game]['block'] = []
        state[self.game]['ball'] = []

        for index, body in enumerate(self.space.bodies):
            shape = list(body.shapes)[0]
            x, y = body.position
            if isinstance(shape, pymunk.Segment):    
                a_x, a_y = shape.a[0], shape.a[1]
                b_x, b_y = shape.b[0], shape.b[1]
                x1, x2 = x + a_x, x + b_x
                y1, y2 = y + a_y, y + b_y
                state[self.game]['block'].append([x1, y1, x2, y2, self.eli[index], self.dynamic[index], body.velocity[0], body.velocity[1], body.angular_velocity, 0, 0])
                    
            elif isinstance(shape, pymunk.Circle):
                state[self.game]['ball'].append([x, y, shape.radius, self.eli[index], self.dynamic[index], body.velocity[0], body.velocity[1], body.angular_velocity, 0, 0])               
                    

        def find_body_index(body, body_list):
            for i, b in enumerate(body_list):
                if b == body:
                    return i
            return None
        
        joint_cnt, spring_cnt = 0, 0
        len_ = len(state[self.game]['block'])
        for constraint in self.space.constraints:
            body_a, body_b = constraint.a, constraint.b
            if isinstance(constraint, pymunk.constraints.PinJoint):
                joint_cnt += 1
                index_a = find_body_index(body_a, self.space.bodies)
                index_b = find_body_index(body_b, self.space.bodies)
                self.add_constraint_to_body('joint', index_a, len_, joint_cnt, state[self.game])
                self.add_constraint_to_body('joint', index_b, len_, joint_cnt, state[self.game])
            elif isinstance(constraint, pymunk.constraints.DampedSpring):
                spring_cnt += 1
                index_a = find_body_index(body_a, self.space.bodies)
                index_b = find_body_index(body_b, self.space.bodies)
                self.add_constraint_to_body('spring', index_a, len_, spring_cnt, state[self.game])
                self.add_constraint_to_body('spring', index_b, len_, spring_cnt, state[self.game])  
        return state[self.game]

    def add_ball(self, b_pos, radius, mass, elasticity, friction):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = b_pos[0], b_pos[1]
        shape = pymunk.Circle(body, radius)
        shape.elasticity = elasticity
        shape.friction = friction
        shape.color = (255, 0, 0, 255)
        self.space.add(body, shape)
        return body

    def add_static_line(self, l_pos, eli, friction, elasticity):
        static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        x1, y1, x2, y2 = l_pos[0][0], l_pos[0][1], l_pos[1][0], l_pos[1][1]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        static_body.position = x, y
        static_shape = pymunk.Segment(static_body, (x1 - x, y1 - y), (x2 - x, y2 - y), 10)
        static_shape.friction = friction
        static_shape.elasticity = elasticity
        if eli:
            static_shape.color = (164, 164, 164, 255)
        else:
            static_shape.color = (0, 0, 0, 255)
        self.space.add(static_body, static_shape)
        return static_body

    def add_dynamic_line(self, l_pos, friction, elasticity):
        x1, y1, x2, y2 = l_pos[0][0], l_pos[0][1], l_pos[1][0], l_pos[1][1]
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        mass = 1.0
        moment = pymunk.moment_for_segment(mass, (0, 0), (0, 0), 10.)
        body = pymunk.Body(mass, moment)
        body.position = x, y
        shape = pymunk.Segment(body, (x1 - x, y1 - y), (x2 - x, y2 - y), 10.)
        shape.friction = friction
        shape.elasticity = elasticity
        self.space.add(body, shape)
        return body

    def add_joint(self):
        for (b1, b2) in self.PARAS[self.game]['joint']:
            c: pymunk.Constraint = pymunk.PinJoint(self.space.bodies[b1], self.space.bodies[b2])
            self.space.add(c)

    def add_spring(self):
        for (b1, b2) in self.PARAS[self.game]['spring']:
            c = pymunk.DampedSpring(self.space.bodies[b1], self.space.bodies[b2], (0, 0), (0, 0), 20, 1, 0.3)
            self.space.add(c)

    def add_all(self):
        assert len(self.blocks) == len(self.PARAS[self.game]['eli'][:-self.num_ball])
        for l_para, eli, dynamics in zip(self.blocks, self.PARAS[self.game]['eli'][:-self.num_ball],
                                         self.PARAS[self.game]['dynamic'][:-self.num_ball]):
            if dynamics:
                self.add_dynamic_line(l_para, self.l_friction, self.l_elasticity)
            else:
                self.add_static_line(l_para, eli, self.l_friction, self.l_elasticity)
        for b_para in self.balls:
            self.add_ball(b_para[:2], b_para[2], self.b_mass, self.b_elasticity, self.b_friction)
        if self.joint:
            self.add_joint()
        if self.spring:
            self.add_spring()

    def eliminate(self, p):
        for i, body in enumerate(self.space.bodies[:-self.num_ball]):
            shape = list(body.shapes)[0]
            x, y = body.position
            shape = list(body.shapes)[0]
            a_x, a_y = shape.a[0], shape.a[1]
            b_x, b_y = shape.b[0], shape.b[1]
            x1, x2 = x + a_x, x + b_x
            y1, y2 = y + a_y, y + b_y
            length = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            lv = np.array([x1 - x2, y1 - y2]) / length
            pv = np.array([p[0] - x, p[1] - y])
            if self.eli[i] == 1 and np.abs(np.dot(pv, lv)) < length /2 and np.abs(np.cross(pv, lv)) < 10:
                self.space.remove(shape, shape.body)
                for constraint in list(shape.body.constraints):
                    self.space.remove(constraint)
                self.eli.pop(i)
                self.eli_mask.pop(i)
                self.dynamic.pop(i)
                self.shape.pop(i)
                return i
        return -1

    def examine_success(self):
        success = 0
        for ball in self.space.bodies[-self.num_ball:]:
            if ball.position[1] > self.HEIGHT:
                success += 1
        if success == self.num_ball:
            return True
        else:
            return False
    
    def get_property_full(self, body, idx, shape_flag):

        """
        For blocks:
            [xl, yl, xr, yr, eli, dyn, vx, vy, w_rotate, spring, joint] 
        For balls:
            [x,  y,  x,  y,  radius,  eli, dyn, vx, vy, w_rotate, spring, joint]
        """
        x, y = body.position
        shape = list(body.shapes)[0]
        if shape_flag:
            r = 10
            a_x, a_y = shape.a[0], shape.a[1]
            b_x, b_y = shape.b[0], shape.b[1]
            x1, x2 = x + a_x, x + b_x
            y1, y2 = y + a_y, y + b_y
            prop = [x1, y1, x2, y2, self.eli[idx], self.dynamic[idx], body.velocity[0], body.velocity[1], body.angular_velocity, 0, 0]
        else:
            r = shape.radius
            prop = [x, y, x, y, r, self.eli[idx], self.dynamic[idx], body.velocity[0], body.velocity[1], body.angular_velocity, 0, 0]
        if 'joint' in self.PARAS[self.game].keys():
            if idx in sum(self.PARAS[self.game]['joint'], []):
                prop[-1] = 1
        if 'spring' in self.PARAS[self.game].keys():
            if idx in sum(self.PARAS[self.game]['spring'], []):
                prop[-2] = 1
        return prop
    
    def get_property(self, body, idx, shape_flag):
        """
        For blocks:
            Given position and a, b
            Return the two end points, radius, eli, dynamic, joint and spring
        For balls:
            Return duplicated center position, radius, eli, dynamic, joint and spring
        """
        x, y = body.position
        shape = list(body.shapes)[0]
        if shape_flag:
            r = 10
            a_x, a_y = shape.a[0], shape.a[1]
            b_x, b_y = shape.b[0], shape.b[1]
            x1, x2 = x + a_x, x + b_x
            y1, y2 = y + a_y, y + b_y
            prop = [x1, y1, x2, y2, r, self.eli[idx], self.dynamic[idx], 0, 0]
        else:
            r = shape.radius
            prop = [x, y, x, y, r, self.eli[idx], self.dynamic[idx], 0, 0]
        if 'joint' in self.PARAS[self.game].keys():
            if idx in sum(self.PARAS[self.game]['joint'], []):
                prop[-2] = 1
        if 'spring' in self.PARAS[self.game].keys():
            if idx in sum(self.PARAS[self.game]['spring'], []):
                prop[-1] = 1
        return prop

    def get_all_property(self, full_property=False):
        # all_property = np.zeros((self.max_obj_num, 9))
        all_property = [[0.] * 9 for _ in range(self.max_obj_num)]
        for i, body in enumerate(self.space.bodies):
            index = self.eli_mask[i]
            if not full_property:
                all_property[index] = self.get_property(body, i, self.shape[i])
            else :
                all_property[index] = self.get_property_full(body, i, self.shape[i]) # each item either with length 11 (for block) or 9 (for ball)
        return all_property

    def object_idx_to_pos(self, idx):
        actions = self.get_action_space_by_property()
        for i, pos_x, pos_y in actions:
            if i == idx:
                return pos_x, pos_y
        raise ValueError('idx not found in action list')
    
    def get_action_space_by_property(self):
        '''
        In the properties array, return the list of (idx, pos_x, pos_y) that can be eliminated, won't return empty action, won't pad.
        '''

        actions = []
        all_property = self.get_all_property()
        for idx, property in enumerate(all_property):
            if property[-4] == 1:
                x = (property[0] + property[2]) / 2
                y = (property[1] + property[3]) / 2
                actions.append((idx, x, y))

        return actions

    def get_action_space(self):
        '''
        Return the central positions of eliminable blocks 
        with no action at the first place and the padding place.
        '''
        self.reset()
        actions = [[0., 0.]] # no action
        all_property = self.get_all_property()
        for property in all_property:
            if property[-4] == 1:
                x = (property[0] + property[2]) / 2
                y = (property[1] + property[3]) / 2
                actions.append([x, y])
        for _ in range(self.max_eli_obj_num + 1 - len(actions)):
            actions.append([0., 0.])
        return actions

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption(f"Interactive Physical Reasoning: {self.game}")
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def reset(self, use_images=False):
        for body in self.space.bodies:
            shape = list(body.shapes)[0]
            self.space.remove(shape, shape.body)
        for joint in self.space.constraints:
            self.space.remove(joint)
        if self.screen:
            self.screen.fill((255, 255, 255))
        self.add_all()
        self.shape = [1] * len(self.blocks) + [0] * len(self.balls)
        self.eli = deepcopy(self.PARAS[self.game]['eli'])
        self.eli_mask = [i for i in range(len(self.space.bodies))]
        self.dynamic = deepcopy(self.PARAS[self.game]['dynamic'])

        self.current_time = 0

        if use_images:
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            image = pygame.surfarray.array3d(self.screen)
            image = image.swapaxes(0,1)
            image[:,:,[0,2]] = image[:,:,[2,0]]
            return image
        else:
            return self.get_all_property()

    def step(self, pos, use_images=False, timestep = None): # pos: the position to be eliminated

        if timestep is None:
            timestep = self.timestep

        reward = self.step_reward
        done = False
        if pos == [0., 0.]:
            self.space.step(timestep)
        else:
            index = self.eliminate(pos)
            self.space.step(timestep)
            if index != -1:
                reward += self.eli_reward

        if self.examine_success():
            reward += self.success_reward
            done = True
        
        self.current_time += timestep

        if use_images:
            self.screen.fill((255, 255, 255))
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            image = pygame.surfarray.array3d(self.screen)
            image = image.swapaxes(0,1)
            image[:,:,[0,2]] = image[:,:,[2,0]]
            return image, reward, done
        else:
            return self.get_all_property(), reward, done

    def check_static(self):
        '''
        Check if the system is static if no action is implemented.
        '''
        copied_space = deepcopy(self.space)
        copied_space.step(self.timestep)
        copied_space.step(self.timestep)

        current_frame_properties = self.get_all_property()

        next_frame_properties = np.zeros((self.max_obj_num, 9))
        for i, body in enumerate(copied_space.bodies):
            index = self.eli_mask[i]
            next_frame_properties[index] = self.get_property(body, i, self.shape[i])
        
        for i in range(len(current_frame_properties)):
            diff = np.linalg.norm(current_frame_properties[i] - next_frame_properties[i])
            #print(f"ori: {current_frame_properties[i]}, next: {next_frame_properties[i]}, diff: {diff}")
            if diff > 1e-8:
                return False

        return True

    def demonstrate_scene(self):
        '''
        For debugging, press 'Q' to quit the demonstration. 
        Mark each object with its index, meanwhile print the property of each object.
        '''
        self.init_screen()
        self.reset()
        print("----- Demonstrate the scene -----")

        font = pygame.font.Font(None, 24)
        running = True

        # Fill the screen with white background
        self.screen.fill((255, 255, 255))

        # Draw the objects in the space
        self.space.debug_draw(self.draw_options)

        properties = self.get_all_property(full_property=True) # need velocity

        if_static = self.check_static()
        print("Is static: ", if_static)

        # Iterate through objects in the space and draw their index
        for i, body in enumerate(self.space.bodies):
            if isinstance(body, pymunk.Body) and body.shapes is not None:
                # Get the position of the object
                pos = body.position

                # Render the index as text
                text_surface = font.render(str(i), True, (128, 0, 128))
                self.screen.blit(text_surface, (pos.x, pos.y))

                # Print the object's properties
                print(f"Object {i}: {properties[i]}")

        # Update the display to show the static scene
        pygame.display.flip()

        # Wait for the user to quit by pressing 'Q'
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

        # pygame.quit()


    def simulate(self, action=[], use_current_time = False):
        '''
        The action consists of many tuples (x, y, t), indicating clicking position (x, y) at time t.
        '''
        action.sort(key=lambda a: a[-1])
        step, valid_step, time_count = 0, 0, 0

        # !
        if use_current_time: time_count = self.current_time

        total_step = len(action)
        while time_count < self.max_time:
            if step < total_step:
                p, t = action[step][0:2], action[step][2]
                if time_count >= t:
                    if t != 0:
                        if self.eliminate(p) != -1:
                            valid_step += 1
                    step += 1

            self.space.step(self.timestep)
            time_count += self.timestep
            if self.examine_success():
                return 1, valid_step, time_count
        return 0, valid_step, time_count

    def simulate_vis(self, action=[]):
        self.init_screen()
        self.reset()
        action.sort(key=lambda a: a[-1])
        step, valid_step, time_count = 0, 0, 0
        total_step = len(action)
        while time_count < self.max_time:
            self.screen.fill((255, 255, 255))
            if step < total_step:
                p, t = action[step][0: 2], action[step][2]
                if time_count >= t:
                    if t != 0:
                        if self.eliminate(p) != -1:
                            print(f'Step {step}: Click {p} at time {time_count}.')
                    step += 1

            self.space.step(self.timestep)
            time_count += self.timestep
            self.space.debug_draw(self.draw_options)
            if self.examine_success():
                print(f'###### Success at time {time_count} ######')
                self.add_text(text="Success!", loc=(230, 30), color="green", font=50)
                pygame.display.flip()
                time.sleep(2)
                return
            pygame.display.flip()
            self.clock.tick(self.FPS)
        self.add_text(text="Failed", loc=(245, 30), color="red", font=50)
        pygame.display.flip()
        time.sleep(2)

    def play_debug(self):

        pygame.font.init()
        font = pygame.font.Font(None, 24)
        
        self.init_screen()
        self.reset()
        finish_game, exceed_time, start = False, False, False
        time_count = 0
        continue_to_next_frame = False

        while time_count < self.max_time + self.timestep:
            self.screen.fill((255, 255, 255))
            
            for event in pygame.event.get():
                    if event.type == QUIT:
                        sys.exit(0)
                    elif event.type == KEYDOWN and event.key == K_q:
                        continue_to_next_frame = True
                    
                    elif event.type == KEYDOWN and event.key == K_n:
                        return
                    elif event.type == KEYDOWN and event.key == K_s:
                        start = True
                    elif event.type == KEYDOWN and event.key == K_r:
                        button_require = True
                        if button_require:
                            self.reset()
                            time_count = 0
                            finish_game, exceed_time, start = False, False, False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        p = event.pos
                        if not finish_game and start:
                            self.eliminate(p)
                    elif event.type == KEYUP:
                        if event.key == K_q:
                            continue_to_next_frame = False  # Reset the flag when Q is released
            if start:
                    if continue_to_next_frame:
                        print("----- Demonstrate the scene -----")
                        time_count += self.timestep
                        print(f'Time: {time_count}')
                        if_static = self.check_static()
                        print("Is static: ", if_static)

                    if time_count >= self.max_time - self.timestep:
                        self.add_text(text="Failed", loc=(245, 30), color="red")
                        time_count = self.max_time
                        exceed_time, finish_game = True, True

                    if not exceed_time and self.examine_success():
                        self.add_text(text="Success!", loc=(230, 30), color="green")
                        time_count = 0
                        finish_game = True
                    if continue_to_next_frame:
                        self.space.step(self.timestep)
                    self.space.debug_draw(self.draw_options)

                    ## mark objects
                    properties = self.get_all_property(full_property=True) # need velocity
                    # Iterate through objects in the space and draw their index
                    for i, body in enumerate(self.space.bodies):
                        if isinstance(body, pymunk.Body) and body.shapes is not None:
                            # Get the position of the object
                            pos = body.position

                            # Render the index as text
                            text_surface = font.render(str(self.eli_mask[i]), True, (128, 0, 128))
                            self.screen.blit(text_surface, (pos.x, pos.y))

                            # Print the object's properties
                            if continue_to_next_frame :
                                print(f"Object {self.eli_mask[i]}: {properties[self.eli_mask[i]]}")

                    ## print actions 
                    if continue_to_next_frame :
                        actions = self.get_action_space_by_property()
                        print(f"actions:\n{actions}")

                    ## check restore_scene() / print states
                    if continue_to_next_frame :
                        states = self.restore_scene()
                        print(f"states:\n{states}\n")

                    pygame.display.flip()
                    self.clock.tick(self.FPS)
            else:
                    self.add_text(text="Press s to start and r to reset.", loc=(170, 0), color="black", font=30)
                    self.add_text(text="Press n to move on and Quit to exit.", loc=(125, 20), color="black", font=30)
                    self.space.debug_draw(self.draw_options)
                    pygame.display.flip()

    def play(self):
        self.init_screen()
        self.reset()
        finish_game, exceed_time, start = False, False, False
        time_count = 0
        while time_count < self.max_time + self.timestep:
            self.screen.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_n:
                    return
                elif event.type == KEYDOWN and event.key == K_s:
                    start = True
                elif event.type == KEYDOWN and event.key == K_r:
                    button_require = True
                    if button_require:
                        self.reset()
                        time_count = 0
                        finish_game, exceed_time, start = False, False, False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    p = event.pos
                    if not finish_game and start:
                        self.eliminate(p)
            if start:
                time_count += self.timestep
                if time_count >= self.max_time - self.timestep:
                    self.add_text(text="Failed", loc=(245, 30), color="red")
                    time_count = self.max_time
                    exceed_time, finish_game = True, True

                if not exceed_time and self.examine_success():
                    self.add_text(text="Success!", loc=(230, 30), color="green")
                    time_count = 0
                    finish_game = True
                self.space.step(self.timestep)
                self.space.debug_draw(self.draw_options)

                pygame.display.flip()
                self.clock.tick(self.FPS)
            else:
                self.add_text(text="Press s to start and r to reset.", loc=(170, 0), color="black", font=30)
                self.add_text(text="Press n to move on and Quit to exit.", loc=(125, 20), color="black", font=30)
                self.space.debug_draw(self.draw_options)
                pygame.display.flip()

    def collect_initial_data(self, save_path='./game_initial_data/'):
        self.init_screen()
        self.reset()
        self.screen.fill((255, 255, 255))
        game_path = save_path + f'{self.game}/'
        if not os.path.exists(game_path):
            os.makedirs(game_path)
        dic = {'balls': np.array(self.balls),
               'blocks': np.array(self.blocks),
               'eli': np.array(self.eli),
               'dynamic': np.array(self.dynamic)}
        np.save(game_path + 'raw.npy', dic)
        vectors = self.get_all_property()
        np.save(game_path + 'vectors.npy', vectors)

        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        pygame.image.save(self.screen, game_path + f'{self.game}.jpg')
    
    def collect_seq_data(self, save_path='./offline_data/', act_lists=[], fps=10):  # maximum fps=60
        '''
        The act_lists consists of some actions. 
        Each action consists of many tuples (x, y, t), indicating clicking position (x, y) at time t.
        '''
        self.init_screen()
        game_path = save_path + f'{self.game}/'
        if not os.path.exists(game_path):
            os.makedirs(game_path)
        dic = {'balls': np.array(self.balls),
               'blocks': np.array(self.blocks),
               'eli': np.array(self.eli),
               'dynamic': np.array(self.dynamic)}
        np.save(game_path + 'raw.npy', dic)
        for i, act_list in enumerate(act_lists):  # the step number of each action can be variant
            act_list.sort(key=lambda a: a[-1])
            self.reset()
            data_path = game_path + f'{i}/'
            if not os.path.exists(data_path):
                img_path = data_path + 'images/'
                os.makedirs(img_path)
                act_pos = np.array([list(a[0:-1]) for a in act_list])
                act_ts = np.array([a[-1] for a in act_list])
                np.save(data_path + 'actions.npy', np.concatenate((act_pos, act_ts.reshape(-1, 1)), axis=-1))
            else:
                continue  # already get the action_data stored
            step, time_count, save_count = 0, 0., 0
            total_step = len(act_list)
            interval = self.FPS / fps
            interval_cal = 0
            vectors = np.zeros((self.max_time * fps, self.max_obj_num, 9))

            while time_count < self.max_time:
                if step < total_step:
                    p, t = act_list[step][0:2], act_list[step][2]
                    if time_count >= t:
                        if t != 0:
                            self.eliminate(p)
                        step += 1

                if interval_cal == interval or interval_cal == 0:
                    interval_cal = 0
                    vectors[save_count] = self.get_all_property()
                    self.space.debug_draw(self.draw_options)
                    pygame.display.flip()
                    pygame.image.save(self.screen, img_path + f'{save_count}.jpg')
                    save_count += 1

                self.screen.fill((255, 255, 255))
                self.space.step(self.timestep)
                interval_cal += 1
                time_count += self.timestep
                if self.examine_success():
                    np.save(data_path + 'vectors.npy', np.array(vectors))
                    break

            np.save(data_path + 'vectors.npy', np.array(vectors))
    
    def collect_while_play(self, player_name='', max_episode=5, save_path='./player_data.json'):
        self.init_screen()
        self.reset()
        time_count, total_reward, actions = 0, 0, []
        status = 0
        reset = True
        speedup = False
        episode = 0
        p = [0., 0.]
        while time_count < self.max_time:
            self.screen.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_n:
                    # if (status == 2 and episode == max_episode) or status == 3:
                        return
                elif event.type == KEYDOWN and event.key == K_s:
                    if reset and episode < max_episode:
                        status = 1
                        p = [0., 0.]
                elif event.type == KEYDOWN and event.key == K_r:
                    reset_require = True
                    if reset_require and status == 2 and episode < max_episode:
                        self.reset()
                        time_count, total_reward, actions = 0, 0, []
                        status = 0
                        reset = True
                        speedup = False
                elif event.type == KEYDOWN and event.key == K_e:
                    if status == 1:
                        speedup = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    p = event.pos
            if speedup:
                p = [0., 0.]
            if status == 0:
                self.add_text(text=f"Press s to start!", loc=(230, 0), color="black", font=30)
                self.add_text(text=f"Press e to end in advance!", loc=(180, 20), color="black", font=30)
            elif status == 1:
                _, reward, done = self.step(p, self.timestep)
                total_reward += reward
                if reward not in [self.step_reward, self.success_reward + self.step_reward]:
                    actions.append(list(p) + [time_count])

                time_count += self.timestep
                if time_count >= self.max_time - self.timestep:
                    info_dict = {"player": player_name, "game": self.game, "episode": episode, "actions": actions, "reward": total_reward}
                    with open(save_path, 'a') as file:
                        file.writelines(json.dumps(info_dict)+'\n')
                    status = 2
                    episode += 1
                    reset = False
                else:
                    if done:
                        info_dict = {"player": player_name, "game": self.game, "episode": episode, "actions": actions, "reward": total_reward}
                        with open(save_path, 'a') as file:
                            file.writelines(json.dumps(info_dict)+'\n')
                        status = 3
                        episode += 1
                        reset = False
            elif status == 2:
                self.add_text(text=f"Score in episode {episode} / {max_episode}: {round(total_reward, 2)}", 
                loc=(120, 0), color="red", font=40)
                if episode < max_episode:
                    self.add_text(text=f"Press r to reset!", loc=(230, 30), color="black", font=30)
                else:
                    self.add_text(text=f"Press n to move on to the next game.",
                    loc=(125, 30), color="black", font=30)
                    
            elif status == 3:
                self.add_text(text=f"Score in episode {episode} / {max_episode}: {round(total_reward, 2)}", 
                loc=(120, 0), color="green", font=40)
                self.add_text(text=f"Press n to move on to the next game.",
                loc=(125, 30), color="black", font=30)
            else:
                pass
            if not (speedup and status == 1):
                self.space.debug_draw(self.draw_options)
                pygame.display.flip()
                self.clock.tick(self.FPS)

    def add_text(self, text="Success!", loc=(230, 30), color="green", font=50):
        font = pygame.font.Font(None, font)
        text = font.render(text, True, pygame.Color(color))
        self.screen.blit(text, loc)
