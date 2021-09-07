import numpy as np
import os
import copy
from env_rl_noise_option import EnvRLNoiseOption

import models.problem_config as pcf
from models.features_utils import StaticFeatures



class BatchEnvRL:
    base_dir = os.path.dirname(os.getcwd())

    def __init__(self, n_envs, n_nodes=50, seed=None,
                 from_file=False, x_path=None, adj_path=None, verbose=False, adaptive=True, single_inst=False, noise_on=True):

        self.adaptive = adaptive
        self.noise_on=noise_on
        self.envs = []
        if from_file:
            n_nodes = 0
            for i in range(n_envs):
                assert os.path.isfile(x_path)
                assert os.path.isfile(adj_path)
                env = EnvRLNoiseOption(from_file=from_file,
                                       x_path=x_path,
                                       adj_path=adj_path,
                                       adaptive=adaptive,
                                       seed=seed,
                                       noise_on=self.noise_on)
                self.envs.append(env)
                if env.n_nodes > n_nodes:
                    n_nodes = env.n_nodes


            self.n_envs = n_envs
            #n_nodes will be the maximum number of nodes in the files
            self.n_nodes = n_nodes
            if verbose:
                print(f'read {self.n_envs} files, with a max number of nodes: {self.n_nodes}')
        else:
            # print('Generating instances on the fly...')
            if single_inst:
                env = EnvRLNoiseOption(n_nodes=n_nodes, seed=seed, verbose=verbose, adaptive=adaptive, noise_on=self.noise_on)
                for i in range(n_envs):
                    self.envs.append(copy.deepcopy(env))
            else:
                for i in range(n_envs):
                    self.envs.append(EnvRLNoiseOption(n_nodes=n_nodes, seed=seed, verbose=verbose, adaptive=adaptive, noise_on=self.noise_on))
            self.n_envs = n_envs
            self.n_nodes = n_nodes
        # print(f'Created {self.n_envs} environments.')

    def get_sim_name(self):
        names = []
        for env in self.envs:
            names.append(env.get_sim_name())
        return names

    def reset(self):
        for env in self.envs:
            env.reset()

    def get_features(self):
        x = np.zeros((self.n_envs, self.n_nodes, 3))
        idx = 0
        for env in self.envs:
            x_ = np.concatenate((self._normalize_features(env.x[:, 1:3]), env.x[:, -2, None]), axis=-1)
            n_ = len(x_)
            z_ = np.zeros((self.n_nodes-n_, 3))
            x_ = np.concatenate((x_, z_), axis=0)
            x[idx] = x_
            idx += 1
        return x

    def get_our_features_old(self):
        x = np.zeros((self.n_envs, self.n_nodes, 11))
        m = np.zeros((self.n_envs, self.n_nodes, self.n_nodes))

        idx = 0
        for env in self.envs:
            inst_features_, max_travel_times_ = env.get_features()#.copy()
            inst_features, max_travel_times = inst_features_.copy(), max_travel_times_.copy()
            inst_features = inst_features[:, 1:]

            Tmax = inst_features[0,pcf.ARRIVAL_TIME_IDX]
            inst_features[:,pcf.CLOSING_TIME_WINDOW_IDX] = np.minimum(inst_features[:,pcf.CLOSING_TIME_WINDOW_IDX], Tmax) # correct Closing time
            inst_features[:,pcf.OPENING_TIME_WINDOW_IDX] = np.minimum(inst_features[:,pcf.OPENING_TIME_WINDOW_IDX], inst_features[:,pcf.CLOSING_TIME_WINDOW_IDX]) # correct opening time

            tw_delta = inst_features[:,pcf.CLOSING_TIME_WINDOW_IDX] - inst_features[:,pcf.OPENING_TIME_WINDOW_IDX] # time window width
            tmax_O_delta = inst_features[0,pcf.ARRIVAL_TIME_IDX] - inst_features[:,pcf.CLOSING_TIME_WINDOW_IDX]  # Tmax-CLOSING_TIME
            tmax_C_delta = inst_features[0,pcf.ARRIVAL_TIME_IDX] - inst_features[:,pcf.OPENING_TIME_WINDOW_IDX] # Tmax-OPENING_TIME

            score_tw_delta_ratio = inst_features[:,pcf.REWARD_IDX] / (1.0+tw_delta) # score/(1+ time window width)
            score_return_time_ratio = inst_features[:,pcf.REWARD_IDX] / (1.0+max_travel_times[0]) # score/(return time)

            inst_features = np.concatenate([inst_features,
                                            np.expand_dims(tw_delta, axis=1),
                                            np.expand_dims(score_tw_delta_ratio, axis=1),
                                            np.expand_dims(tmax_O_delta, axis=1),
                                            np.expand_dims(tmax_C_delta, axis=1),
                                            np.expand_dims(score_return_time_ratio, axis=1),
                                            ], axis=1)

            x[idx] = inst_features
            m[idx] = max_travel_times

            idx += 1
        return x, m

    def get_our_features(self, feature_list):

        n_features = len(feature_list)
        x = np.zeros((self.n_envs, self.n_nodes, n_features))
        m = np.zeros((self.n_envs, self.n_nodes, self.n_nodes))

        idx = 0
        for env in self.envs:
            inst_features_raw, max_travel_times = env.get_features()

            sf = StaticFeatures(inst_features_raw, max_travel_times)

            x[idx] = sf.compute_inst_features(feature_list)
            m[idx] = max_travel_times

            idx += 1
        return x, m

    @staticmethod
    def _normalize_features(x):
        max_x = np.max(x, axis=0)
        min_x = np.min(x, axis=0)

        return (x - min_x) / (max_x - min_x)

    def step(self, next_nodes):
        tour_time = np.zeros((self.n_envs, 1))
        time_t = np.zeros((self.n_envs, 1))
        rwd_t = np.zeros((self.n_envs, 1))
        pen_t = np.zeros((self.n_envs, 1))
        feas = np.ones((self.n_envs, 1), dtype=bool)
        violation_t = np.zeros((self.n_envs, 1))
        done_t = np.zeros((self.n_envs, 1))

        idx = 0
        for env in self.envs:
            tour_time[idx], time_t[idx], rwd_t[idx], pen_t[idx], feas[idx], violation_t[idx], done_t[idx] = env.step(
                next_nodes[idx][0])
            idx += 1
        return tour_time, time_t, rwd_t, pen_t, feas, violation_t, done_t

    def check_solution(self, sols):
        sols = sols.cpu().detach().numpy()
        if self.adaptive:
            rwds = np.zeros((self.n_envs, 1))
            pens = np.zeros((self.n_envs, 1))
            idx = 0
            for env in self.envs:
                rwds[idx], pens[idx]= env.get_collected_rewards(), env.get_incurred_penalties()
                idx += 1
        else:
            rwds = np.zeros((self.n_envs, 1))
            pens = np.zeros((self.n_envs, 1))
            idx = 0
            for env in self.envs:
                _, rwds[idx], pens[idx], _ = env.check_solution(sols[idx])
                idx += 1
        return rwds, pens

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.getcwd())
    # x_dir = os.path.join(base_dir, 'data', 'valid', 'instances')
    # adj_dir = os.path.join(base_dir, 'data', 'valid', 'adjs')
    x_dir = os.path.join(base_dir, 'data', 'test', 'instances')
    adj_dir = os.path.join(base_dir, 'data', 'test', 'adjs')
    batch_env = BatchEnvRL(n_envs=2, from_file=True, x_dir=x_dir, adj_dir=adj_dir)
