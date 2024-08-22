import numpy as np
import random
from random import sample

from iphyre.simulator import IPHYRE
from iphyre.games import GAMES, PARAS
import math 

def generate_blocks(num_blocks, block_length, slope_range, screen_width = 600, screen_height = 600, padding = 75): 
    blocks = []
    for _ in range(num_blocks):
        left_X = random.randint(padding, screen_width - padding)
        left_Y = random.randint(padding, int(screen_height * 0.85) - padding)
        angle = random.uniform(-slope_range, slope_range)
        angle_rad = math.radians(angle)

        right_X = int(left_X + block_length * math.cos(angle_rad))
        right_Y = int(left_Y + block_length * math.sin(angle_rad))

        if right_X < left_X:
            left_X, right_X = right_X, left_X
            left_Y, right_Y = right_Y, left_Y

        blocks.append([[left_X, left_Y], [right_X, right_Y]])
    
    return blocks

def generate_ball(num_balls = 1, padding = 75, screen_width = 600, screen_height = 600):
    balls = []
    for _ in range(num_balls):
        x = random.randint(padding * 2, screen_width - padding * 2)
        y = random.randint(padding, int(screen_height * 0.5) - padding)
        radius = 20
        balls.append([x, y, radius])
    return balls
    
if __name__ == "__main__":

    random.seed(42)
    np.random.seed(42)

    num_games = int(input("Enter the number of games: "))
    num_eli_objects = int(input("Enter the number of eliminable objects: "))
    num_non_eli_objects = int(input("Enter the number of non-eliminable objects: "))
    block_length = 100
    slope_range = 45 

    settings = {}

    for i in range(num_games):
        name = str(i)
        eli_vector = [1] * num_eli_objects + [0] * num_non_eli_objects + [0]
        dyn_vector = [0] * num_eli_objects + [0] * num_non_eli_objects + [1]
        block_dict = generate_blocks(num_eli_objects + num_non_eli_objects ,block_length, slope_range)
        ball_dict = generate_ball()
        settings[name] = {'eli': eli_vector, 'dynamic': dyn_vector, 'block': block_dict, 'ball': ball_dict}
        print(settings[name])
    names = list(settings.keys())

    for name in names:
        demo = IPHYRE(name, 60, src=settings)
        demo.collect_while_play(player_name="nobody", max_episode=3)
