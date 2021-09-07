import os
from tqdm import tqdm
import numpy as np
import argparse
from time import sleep

from generator.op.instances import InstanceGenerator
import multiprocessing as mp
from multiprocessing import Process

class InstanceGeneratorPlus(InstanceGenerator):
    """For pre-generating instances for training"""

    def save_optw(self, i, inst, adj, x_dir='instances', adj_dir='adjs'):
        path_x = self.make_dir(x_dir)
        path_adj = self.make_dir(adj_dir)

        inst.to_csv(os.path.join(path_x, f'instance_nnodes_{self.n_nodes}_seed_{self.seed}.csv'), index=False, sep=',')
        adj.to_csv(os.path.join(path_adj, f'adj-instance_nnodes_{self.n_nodes}_seed_{self.seed}.csv'), index=False, sep=',')

def setup_args_parser():
    parser = argparse.ArgumentParser(description='generate instances')
    parser.add_argument('--debug', help='', action='store_true')

    return parser

def create_instance(info):
    idx, n_nodes, seed, data_dir = info
    try:
        gen = InstanceGeneratorPlus(1, n_nodes=n_nodes,
                                    seed=seed,
                                    data_dir=data_dir)
        gen.generate_instance_files()
    except Exception as e:
        print (idx)
        print (e)
        sleep(0.01)
    return

if __name__ == '__main__':
    parser = setup_args_parser()
    args = parser.parse_args()

    print (args.debug)

    n_nodes_lb = 10
    n_nodes_ub = 200
    n_nodes_step = 10
    n_nodes_list_ = range(n_nodes_lb, n_nodes_ub+n_nodes_step, n_nodes_step)
    # 200, 190, 180, ..., 10
    n_nodes_list = [n+i for i in range(n_nodes_step+1) for n in n_nodes_list_[::-1]]
    n_nodes_x0 = 4000
    n_nodes_slope = 0
    n_nodes_amount = [n_nodes_x0+n_nodes_slope*i for i, x in enumerate(n_nodes_list)]

    #to exclude already generated
    exclude_n_nodes = [20]

    if args.debug:
        #this is a rough fit to time generate an instance for each n_node
        n_nodes_time_sec = [0.00017*x**2 for x in n_nodes_list]
        n_nodes_time_sec_total = [n*t for n,t in zip(n_nodes_amount, n_nodes_time_sec)]

        print (f'number of different n_nodes: {len(n_nodes_list)}')
        print ('')
        print (f'total amount of instances: {sum(n_nodes_amount)}')
        print ('n_nodes', 'amounts')
        print (np.array(list([x,y] for x,y in zip(n_nodes_list, n_nodes_amount))))
        print ('')
        # print ('n_nodes', 'time per instance')
        # print (np.array(list([round(s, 3)] for s in (n_nodes_time_sec))))
        # print ('')
        total_hours = sum(n_nodes_time_sec_total)/60/60
        max_hours = max(n_nodes_time_sec_total)/60/60
        print ('')
        print (f'{round(total_hours, 2)} total hours')
        print (f'{round(max_hours, 2)} hours for longest n_nodes')

    else:
        print("Number of cpu : ", mp.cpu_count())
        # instantiating process with arguments
        pool = mp.Pool(mp.cpu_count())
        argument_list = []
        for idx, n_nodes in enumerate(n_nodes_list):
           if n_nodes not in exclude_n_nodes:
                data_dir = f'data/generated/n_nodes_{n_nodes}'

                seed_lb, seed_ub = 1, n_nodes_amount[idx]
                seeds = range(seed_lb, seed_ub+1)
                #print (f'generating {len(seeds)} instances with n_nodes {n_nodes} for seeds {seed_lb} to {seed_ub}' )
                for seed in seeds:
                    argument_list.append((idx, n_nodes, seed, data_dir))

    for result in tqdm(pool.imap(func=create_instance, iterable=argument_list), total=len(argument_list)):
        x=1
