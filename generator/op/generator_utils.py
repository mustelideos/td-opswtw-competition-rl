import os
from os import path
import numpy as np


def get_generated_seeds():
    base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    gen_seeds = dict()
    for n_nodes in range(10, 211):
        all_inst = os.listdir(f'{base_dir}/data/generated/n_nodes_{n_nodes}/instances/')
        gen_seeds[n_nodes] = np.sort([int(s.split('_')[4].split('.')[0]) for s in all_inst if '.csv' in s])
    return gen_seeds

if __name__ == '__main__':
    print (get_generated_seeds())
