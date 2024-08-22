'''
    Transform the original game scene data into a format that can be directly input to neural network.
    The format looks like:
    'support':{
        'block' : [
            [xl, yl, xr, yr, eli, dyn, vx, vy, w_rotate, spring, joint] (where spring and joint are 1,2,...)
            ...
        ],
        'ball' : [
            [x,  y, radius,  eli, dyn, vx, vy, w_rotate, spring, joint] (where spring and joint are 1,2,...)
            ...
        ]
    }
'''

from iphyre.games import PARAS, GAMES
import json 


def custom_format(obj):
    if isinstance(obj, dict):
        return {k: custom_format(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return json.dumps(obj)
    else:
        return obj
    
if __name__ == '__main__':

    dict_path = '../data/firstFrameScene.json'

    res = {}

    vx, vy, w_rotate = 0., 0., 0.

    for game in GAMES:
        raw_data = PARAS[game]
        block = []
        ball = []
        block_num = len(raw_data['block'])
        for i in range(block_num):
            xl, yl = raw_data['block'][i][0]
            xr, yr = raw_data['block'][i][1]
            eli = raw_data['eli'][i]
            dyn = raw_data['dynamic'][i]
            block.append([xl, yl, xr, yr, eli, dyn, vx, vy, w_rotate, 0, 0])
        ball_num = len(raw_data['ball'])
        for i in range(ball_num):
            x, y, radius = raw_data['ball'][i]
            eli = raw_data['eli'][i + block_num]
            dyn = raw_data['dynamic'][i + block_num]
            ball.append([x, y, radius, eli, dyn, vx, vy, w_rotate, 0, 0])

        if 'spring' in raw_data:
            for i, (id1, id2) in enumerate(raw_data['spring']):
                if (id1 < block_num):
                    block[id1][-2] = i + 1
                else :
                    ball[id1 - block_num][-2] = i + 1
                if (id2 < block_num):
                    block[id2][-2] = i + 1
                else :
                    ball[id2 - block_num][-2] = i + 1

        if 'joint' in raw_data:
            for i, (id1, id2) in enumerate(raw_data['joint']):
                if (id1 < block_num):
                    block[id1][-1] = i + 1
                else :
                    ball[id1 - block_num][-1] = i + 1
                if (id2 < block_num):
                    block[id2][-1] = i + 1
                else :
                    ball[id2 - block_num][-1] = i + 1
        
        res[game] = {'block': block, 'ball': ball}
    
    formatted_data = custom_format(res)
    with open(f'{dict_path}', 'w') as f:
        json.dump(formatted_data, f, indent=4)
