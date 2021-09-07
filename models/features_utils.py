import torch
import models.problem_config as pcf
import numpy as np
import pickle


class Scaler:
    def __init__(self):

        self.stats_dict = pickle.load(open("../data/generated/normalizations_stats_per_n_node.pkl", "rb" ))

    def data_scaler(self, data, feature_list, n_nodes):


        datan = data.clone()
        for k, feature in enumerate(feature_list):

            if feature == 'x_coordinate':
                datan[:, :, k]/= 200.
            elif feature == 'y_coordinate':
                datan[:, :, k] /= 50.
            elif feature == 'n_nodes':
                datan[:, :, k] /= 200.
            else:
                datan[:,:, k] = (datan[:,:, k] - self.stats_dict[feature][n_nodes]['mean'])/ self.stats_dict[feature][n_nodes]['std']

        return datan

class ScalerGlob:
    def __init__(self):

        self.stats_dict = pickle.load(open("../data/generated/normalizations_stats.pkl", "rb" ))

    def data_scaler(self, data, feature_list, n_nodes):


        datan = data.clone()
        for k, feature in enumerate(feature_list):

            if feature == 'x_coordinate':
                datan[:, :, k]/= 200.
            elif feature == 'y_coordinate':
                datan[:, :, k] /= 50.
            elif feature == 'n_nodes':
                datan[:, :, k] /= 200.
            else:
                datan[:,:, k] = (datan[:,:, k] - self.stats_dict[feature]['mean'])/ self.stats_dict[feature]['std']

        return datan

class StaticFeatures():
    POSSIBLE_FEATURES = ['x_coordinate', 'y_coordinate', 'tw_low',
                         'tw_high', 'prize', 'tmax', 'tw_delta',
                         'tw_low_tmax_delta', 'tw_high_tmax_delta',
                         'prize_tw_delta_ratio',
                         'prize_max_return_time_ratio',
                         'tw_low_tmax_ratio',
                         'tw_high_tmax_ratio', 'n_nodes']

    def __init__(self, instance_features, max_travel_times):
        super(StaticFeatures, self).__init__()
        self.inst_feat = instance_features.copy()
        self.max_travel_times = max_travel_times.copy()
        self.features = list()
        self.feature_list = None
        self.possible_features = self.POSSIBLE_FEATURES

        self._remove_customer_number()
        self._clean_opening_and_closing_times()


    def _remove_customer_number(self):
        self.inst_feat = self.inst_feat[:, 1:]

    def _clean_opening_and_closing_times(self):
        """ correct opening and closing times, i.e. O <= C <= Tmax"""
        tmax = self.inst_feat[0, pcf.TMAX_IDX]
        self.inst_feat[:, pcf.C_IDX] = np.minimum(self.inst_feat[:, pcf.C_IDX], tmax)
        self.inst_feat[:, pcf.O_IDX] = np.minimum(self.inst_feat[:, pcf.O_IDX],
                                                  self.inst_feat[:, pcf.C_IDX])
    def get_feat_x_coordinate(self):
        return self.inst_feat[:, pcf.X_IDX]

    def get_feat_y_coordinate(self):
        return self.inst_feat[:, pcf.Y_IDX]

    def get_feat_tw_low(self):
        return self.inst_feat[:, pcf.O_IDX]

    def get_feat_tw_high(self):
        return self.inst_feat[:, pcf.C_IDX]

    def get_feat_prize(self):
        return self.inst_feat[:, pcf.PRIZE_IDX]

    def get_feat_tmax(self):
        return self.inst_feat[:, pcf.TMAX_IDX]

    def get_feat_tw_low_tmax_ratio(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            tw_low_tmax_ratio = self.inst_feat[:, pcf.O_IDX]/ \
                                self.inst_feat[0, pcf.TMAX_IDX]
        return np.nan_to_num(tw_low_tmax_ratio,
                                nan=0.0, posinf=0.0, neginf=0.0)


    def get_feat_tw_high_tmax_ratio(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            tw_high_tmax_ratio = self.inst_feat[:, pcf.C_IDX]/ \
                                self.inst_feat[0, pcf.TMAX_IDX]
        return np.nan_to_num(tw_high_tmax_ratio,
                                nan=0.0, posinf=0.0, neginf=0.0)

    def get_feat_tw_delta(self):
        """time window width: C-O"""
        return self.inst_feat[:, pcf.C_IDX] - self.inst_feat[:, pcf.O_IDX]

    def get_feat_tw_high_tmax_delta(self):
        """Tmax-CLOSING_TIME"""
        return self.inst_feat[0, pcf.TMAX_IDX] - self.inst_feat[:, pcf.C_IDX]

    def get_feat_tw_low_tmax_delta(self):
        """Tmax-OPENING_TIME"""
        return self.inst_feat[0, pcf.TMAX_IDX] - self.inst_feat[:, pcf.O_IDX]

    def get_feat_n_nodes(self):
        """number of nodes"""
        return self.inst_feat.shape[0] * np.ones((self.inst_feat.shape[0]))

    def get_feat_prize_tw_delta_ratio(self):
        """prize/(1+time window width)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            tw_delta = self.inst_feat[:, pcf.C_IDX] - self.inst_feat[:, pcf.O_IDX]
            prize_tw_delta_ratio = self.inst_feat[:, pcf.PRIZE_IDX] / (tw_delta)
        return np.nan_to_num(prize_tw_delta_ratio,
                                nan=0.0, posinf=0.0, neginf=0.0)

    def get_feat_prize_max_return_time_ratio(self):
        """prize/(return time)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            prize_max_return_time_ratio = self.inst_feat[:, pcf.PRIZE_IDX]\
                    / (self.max_travel_times[pcf.DEPOT_IDX])
        return np.nan_to_num(prize_max_return_time_ratio,
                                nan=0.0, posinf=0.0, neginf=0.0)


    def _concat_features(self):
        return np.concatenate(\
                [np.expand_dims(f, axis=1) for f in self.features],
                              axis=1)

    def compute_inst_features(self, feature_list):

        self.feature_list = feature_list
        undefined_feat = set(self.feature_list)-set(self.possible_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_features)}'
        assert len(undefined_feat)==0, assert_msg

        for f in self.feature_list:
            self.features.append(eval(f'self.get_feat_{f}()'))
        return self._concat_features()



class DynamicFeatures():

    def __init__(self, args):
        super(DynamicFeatures, self).__init__()

        self.arrival_time_idx = pcf.ARRIVAL_TIME_IDX
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.reward_idx = pcf.REWARD_IDX
        self.depot_idx = pcf.DEPOT_IDX

        self.tmax_idx = pcf.TMAX_IDX
        self.o_idx = pcf.O_IDX
        self.c_idx = pcf.C_IDX
        self.prize_idx = pcf.PRIZE_IDX
        self.depot_idx = pcf.DEPOT_IDX

        self.device = args.device

        self.num_dyn_feat = args.ndfeatures

    def make_dynamic_feat(self, data, step, reward_seq, penalty_seq, current_time, current_poi_idx, dist_mat, batch_idx):

        _ , n_nodes, input_size  = data.size()
        batch_size = batch_idx.shape[0]

        dyn_feat = torch.ones(batch_size, n_nodes, self.num_dyn_feat).to(self.device)

        tmax = data[:, self.depot_idx, self.tmax_idx].unsqueeze(1)

        arrive_j_times = current_time + dist_mat[batch_idx, current_poi_idx, :]
        time2depot = dist_mat[batch_idx, self.depot_idx, :]

        dyn_feat[:,:,0] = (data[batch_idx, :, self.o_idx] - current_time) \
                             / tmax
        dyn_feat[:,:,1] = (data[batch_idx, :, self.c_idx] - current_time) \
                             / tmax
        dyn_feat[:,:,2] = (data[batch_idx, :, self.tmax_idx] - current_time) \
                             / tmax

        dyn_feat[:,:,3] = current_time / tmax

        dyn_feat[:,:,4] = (arrive_j_times) / tmax
        dyn_feat[:,:,5] = (data[batch_idx, :, self.o_idx] - arrive_j_times) \
                             / tmax
        dyn_feat[:,:,6] = (data[batch_idx, :, self.c_idx] - arrive_j_times) \
                             / tmax
        dyn_feat[:,:,7] = (data[batch_idx, :, self.tmax_idx] - arrive_j_times) \
                             / tmax

        # ---------------- new features ---------------
        # waiting time
        waitj = torch.max(torch.zeros(batch_size, 1).to(self.device),
                          data[batch_idx, :, self.o_idx]-arrive_j_times)

        c1 = arrive_j_times + waitj <= data[batch_idx, :, self.c_idx] # hard closing time condition
        c2 = arrive_j_times + waitj + time2depot <= tmax # hard return  condition
        c3 = arrive_j_times <= tmax # arrive before max tour duration

        dyn_feat[:,:,8] = waitj / tmax
        dyn_feat[:,:,9] = (waitj + arrive_j_times) / tmax

        dyn_feat[:,:,10] = c1.to(torch.float32)
        dyn_feat[:,:,11] = c2.to(torch.float32)
        dyn_feat[:,:,12] = (c1*c2).to(torch.float32) # c1 and c2

        reward = data[batch_idx, :, self.prize_idx]

        # reward/(norm time to location)
        nr1 =  reward / ((arrive_j_times + waitj) / tmax)
        nr1[(arrive_j_times + waitj).eq(0)]= -1
        dyn_feat[:,:,13] = nr1

        # reward/(norm time to location + time to end location)
        nr2 = reward / ((arrive_j_times + waitj + time2depot) / tmax)
        nr2[(arrive_j_times + time2depot).eq(0)]= -1
        dyn_feat[:,:,14] = nr2

        # reward/(norm time to close)
        nr3 = reward / ((data[batch_idx, :, self.c_idx] - arrive_j_times)/ tmax)
        nr3[(data[batch_idx, :, self.c_idx] - arrive_j_times).eq(0)]= -1
        dyn_feat[:,:,15] = nr3

        #--------------feasibility conditions assuming best travel times possible ----------------
        best_time_frac = 0.01
        best_arrive_j_times =  current_time + \
                        best_time_frac*dist_mat[batch_idx, current_poi_idx, :]

        # waiting time assuming best travel times possible
        best_chance_waitj = torch.max(torch.zeros(batch_size, 1).to(self.device),
                                      data[batch_idx, :, self.o_idx]-best_arrive_j_times)

        # closing time condition assuming best travel times possible
        c4 = best_arrive_j_times + best_chance_waitj <= data[batch_idx, :, self.c_idx]
        # return condition assuming best travel times possible
        c5 = best_arrive_j_times + best_chance_waitj + best_time_frac*time2depot <= tmax

        dyn_feat[:,:,16] = best_chance_waitj / tmax
        dyn_feat[:,:,17] = (best_arrive_j_times + best_chance_waitj) / tmax
        dyn_feat[:,:,18] = c4.to(torch.float32)
        dyn_feat[:,:,19] = c5.to(torch.float32)
        dyn_feat[:,:,20] = (c4*c5).to(torch.float32)

        # reward/(norm best time to location)
        nr4 =  reward / ((best_arrive_j_times + best_chance_waitj) / tmax)
        nr4[(best_arrive_j_times + best_chance_waitj).eq(0)]= -1
        dyn_feat[:,:,21] = nr4

        # reward/(norm best time to location + best time to end location)
        nr5 = reward / ((best_arrive_j_times + best_chance_waitj + best_time_frac*time2depot) / tmax)
        nr5[(best_arrive_j_times + best_time_frac*time2depot).eq(0)]= -1
        dyn_feat[:,:,22] = nr5

        # reward/(norm time to close)
        nr6 = reward / ((data[batch_idx, :, self.c_idx] - best_arrive_j_times)/ tmax)
        nr6[(data[batch_idx, :, self.c_idx] - best_arrive_j_times).eq(0)]= -1
        dyn_feat[:,:,23] = nr6

        #------------------------Probability features ------------------------------------------
        # first condition
        # time after closing
        time_after_closing = torch.max(torch.zeros(batch_size, 1).to(self.device),
                                       arrive_j_times - data[batch_idx, :, self.c_idx])

        # prob of arriving after j-closing time
        delta = arrive_j_times - current_time
        prob_after_close =  time_after_closing / delta
        prob_after_close[delta.eq(0)] = 0.0

        expected_reward_close = (1.0-prob_after_close) * reward + \
                                prob_after_close * (-1)*torch.ones(batch_size, 1).to(self.device)

        # second condition
        time_after_tmax_j = torch.max(torch.zeros(batch_size, 1).to(self.device),
                                       arrive_j_times - tmax)

        # prob of arriving at j after Tmax
        prob_after_tmax_j =  time_after_tmax_j / delta
        prob_after_tmax_j[delta.eq(0)] = 0.0

        expected_reward_tmax_j = (1.0-prob_after_tmax_j) * reward + \
                                prob_after_tmax_j * (-1) * n_nodes *torch.ones(batch_size, 1).to(self.device)

        expected_reward_j = (1.0-prob_after_close) * (1.0-prob_after_tmax_j) * reward \
                        + (1.0-prob_after_close) * prob_after_tmax_j * (-1) * n_nodes * torch.ones(batch_size, 1).to(self.device) \
                        + (1.0-prob_after_tmax_j) * prob_after_close *  (-1 ) * torch.ones(batch_size, 1).to(self.device)  \
                        + prob_after_close * prob_after_tmax_j * (-1) * n_nodes * torch.ones(batch_size, 1).to(self.device)

        # third condition
        # approx time after end
        time_after_end = torch.max(torch.zeros(batch_size, 1).to(self.device),
                                   (arrive_j_times + waitj + time2depot) - tmax)

        prob_after_end = time_after_end / (delta + waitj + time2depot)
        prob_after_end[(delta + waitj + time2depot).eq(0)] = 0.0

        approx_expected_reward_end = (1.0-prob_after_end) * reward + \
                               prob_after_end * (-1)* n_nodes * torch.ones(batch_size, 1).to(self.device)
        approx_expected_reward_end[approx_expected_reward_end.le(-1.1)] = -1.1

        approx_expected_reward_return = (1.0-prob_after_close) * (1.0-prob_after_end) * reward \
                        + (1.0-prob_after_close) * prob_after_end * (-1) * n_nodes * torch.ones(batch_size, 1).to(self.device) \
                        + (1.0-prob_after_end) * prob_after_close *  (-1 ) * torch.ones(batch_size, 1).to(self.device)  \
                        + prob_after_close * prob_after_end * (-1) * n_nodes * torch.ones(batch_size, 1).to(self.device)
        approx_expected_reward_return[approx_expected_reward_return.le(-1.2)] = -1.2

        # -----------------------------------------
        dyn_feat[:, :, 24] = time_after_closing / tmax
        dyn_feat[:, :, 25] = prob_after_close
        dyn_feat[:, :, 26] = expected_reward_close

        dyn_feat[:, :, 27] = time_after_end / tmax
        dyn_feat[:, :, 28] = prob_after_end
        dyn_feat[:, :, 29] = approx_expected_reward_end
        dyn_feat[:, :, 30] = approx_expected_reward_return
        dyn_feat[:, :, 31] = c3.to(torch.float32)

        dyn_feat[:, :, 32] = time_after_tmax_j / tmax
        dyn_feat[:, :, 33] = prob_after_tmax_j

        return dyn_feat
