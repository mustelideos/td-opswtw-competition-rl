import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributions import Categorical

import sys

import models.problem_config as pcf

ourlogzero = sys.float_info.min

class Lookahead():
    def __init__(self, device):
        super(Lookahead, self).__init__()

        self.device = device
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.tmax = pcf.ARRIVAL_TIME_IDX
        self.depot_idx = pcf.DEPOT_IDX

    def get_adjacency_mask(self, raw_inputs, mask_, dist_mat, pres_act, present_time, batch_idx):
        # feasible neighborhood for each node
        mask = mask_.clone()
        step_batch_size, npoints = mask.shape

        best_time_frac = 1.0

        #one step forward update
        arrive_j = best_time_frac*dist_mat[batch_idx, pres_act] + present_time
        # waiting time assuming best travel times possible

        farrive_j = arrive_j.view(step_batch_size, npoints)
        tw_start = raw_inputs[:, :, self.opening_time_window_idx]
        wait_j = torch.max(torch.zeros(step_batch_size, 1).to(self.device), tw_start-farrive_j)

        fpresent_time = farrive_j + wait_j

        # feasible neighborhood for each node
        adj_mask = mask.unsqueeze(1).repeat(1, npoints, 1)
        arrive_j = best_time_frac*dist_mat[batch_idx] + fpresent_time.unsqueeze(2)

        wait_j = torch.max(torch.zeros(step_batch_size, 1, 1).to(self.device), tw_start.unsqueeze(2)-arrive_j)

        tw_end = raw_inputs[:, :, self.closing_time_window_idx]
        ttime = raw_inputs[:, 0, self.tmax]

        dlast =  best_time_frac*dist_mat[batch_idx, :,  self.depot_idx]

        c1 = arrive_j + wait_j <= tw_end.unsqueeze(1)
        c2 = arrive_j + wait_j + dlast.unsqueeze(1) <= ttime.unsqueeze(1).unsqueeze(1).expand(-1, npoints, npoints)
        adj_mask = adj_mask * c1 * c2

        # self-loop
        idx = torch.arange(0, npoints, device=self.device).expand(step_batch_size, -1)
        adj_mask[:, idx, idx] = 1

        return adj_mask


class RunEpisode(nn.Module):
    def __init__(self, neuralnet, args, dynamic_features, use_lookahead=False):
        super(RunEpisode, self).__init__()

        self.device = args.device
        self.neuralnet = neuralnet
        self.dyn_feat = dynamic_features(args)

        self.use_lookahead = use_lookahead
        self.lookahead = Lookahead(self.device)

        self.depot_idx = pcf.DEPOT_IDX
        self.arrival_time_idx = pcf.ARRIVAL_TIME_IDX
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.reward_idx = pcf.REWARD_IDX

    def forward(self, enviroment, inst_features, inst_features_scaled, dist_mat, infer_type):

        self.batch_size, sequence_size, input_size = inst_features.size()

        h_0, c_0 = self.neuralnet.decoder.hidden_0

        dec_hidden = (h_0.expand(self.batch_size, -1), c_0.expand(self.batch_size, -1))

        mask = torch.ones(self.batch_size, sequence_size, device=self.device, requires_grad=False, dtype = torch.uint8)

        present_time = torch.zeros(self.batch_size, 1, device=self.device)

        llog_probs, lactions, lseq_mask, lentropy = [], [], [], []
        reward_seq = torch.zeros(self.batch_size, 1, device=self.device)
        penalty_seq = torch.zeros(self.batch_size, 1, device=self.device)

        actions = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)

        batch_idx = torch.arange(0, self.batch_size, requires_grad=False, device=self.device)

        if self.use_lookahead:
            adj_mask = self.lookahead.get_adjacency_mask(inst_features, mask, dist_mat, actions, present_time, batch_idx)
        else:
            adj_mask = mask
        # encoder first forward pass
        step = 1
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inst_features, step, reward_seq, penalty_seq, present_time, actions, dist_mat, batch_idx)
        emb1 = self.neuralnet.sta_emb(inst_features_scaled)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1,emb2), dim=2)

        _, _, enc_outputs = self.neuralnet(enc_inputs, enc_inputs, adj_mask, enc_inputs, dec_hidden, mask, first_step=True)

        decoder_input = enc_outputs[batch_idx, actions]

        nodes = actions.unsqueeze(1)+1
        lactions.append(nodes)

        # Starting the trip
        is_end_seq = torch.zeros((self.batch_size), requires_grad=False, dtype=torch.bool).to(self.device)
        while not all(is_end_seq):

            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

            actions, log_prob, entropy = self.select_actions(policy, infer_type)

            nodes = actions.unsqueeze(1)+1

            total_time, time, prize, penalty, feasible, violation, env_done = enviroment.step(nodes)

            reward_seq += torch.from_numpy(prize).to(self.device)
            penalty_seq += torch.from_numpy(penalty).to(self.device)

            env_done = torch.from_numpy(env_done).to(self.device)
            present_time = torch.from_numpy(total_time).to(self.device)

            decoder_input = enc_outputs[batch_idx, actions]
            mask = self.update_mask(mask, batch_idx, actions)

            if self.use_lookahead:
                adj_mask = self.lookahead.get_adjacency_mask(inst_features, mask, dist_mat, actions, present_time, batch_idx)
            else:
                adj_mask = mask

            llog_probs.append(log_prob.unsqueeze(1))
            lactions.append(nodes)
            lseq_mask.append(~is_end_seq.unsqueeze(1))
            lentropy.append(entropy.unsqueeze(1))

            step = +1
            bdyn_inputs = self.dyn_feat.make_dynamic_feat(inst_features, step, reward_seq, penalty_seq, present_time, actions, dist_mat, batch_idx)
            emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
            enc_inputs = torch.cat((emb1, emb2), dim=2)

            is_end_seq = (is_end_seq | env_done.eq(1).squeeze(1))

        return torch.cat(lactions, dim=1), torch.cat(llog_probs, dim=1), torch.cat(lentropy, dim=1), torch.cat(lseq_mask, dim=1)

    def update_mask(self, mask , batch_idx, actions):
        nmask = mask.clone()
        nmask[batch_idx, actions] = False
        return nmask

    def select_actions(self, policy, infer_type):
        if infer_type == 'stochastic':
            m = Categorical(policy)
            act_ind = m.sample()
            log_select =  m.log_prob(act_ind)
            poli_entro = m.entropy()
        elif infer_type == 'greedy':
            prob, act_ind = torch.max(policy, 1)
            log_select =  prob.log()
            poli_entro =  torch.zeros(self.batch_size, requires_grad=False).to(self.device)

        return act_ind, log_select, poli_entro
