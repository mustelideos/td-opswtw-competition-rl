import torch
from torch import optim
import torch.nn as nn
from tqdm.notebook import tqdm
import json
import numpy as np
from models.batch_env_rl import BatchEnvRL
# from models.features_utils import Scaler
import copy
from torch.optim.swa_utils import AveragedModel

def train_model(run_episode, data_scaler, opt, scheduler, args, gen_seeds):

    opt.zero_grad()
    n_sims = args.n_sims
    feature_list = args.feature_list
    if args.from_file:
        # to sample uniformly:
        n_nodes = np.random.choice(list(args.n_nodes_list))
        seed = np.random.choice(gen_seeds[n_nodes])
        env = BatchEnvRL(n_envs=args.batch_size, from_file=True, seed=seed,
        x_path=f'../data/generated/n_nodes_{n_nodes}/instances/instance_nnodes_{n_nodes}_seed_{seed}.csv',
        adj_path=f'../data/generated/n_nodes_{n_nodes}/adjs/adj-instance_nnodes_{n_nodes}_seed_{seed}.csv',
        noise_on=args.noise_on)
    else:
        n_nodes = np.random.choice(list(args.n_nodes_list))
        seed = np.random.choice(gen_seeds[n_nodes])
        env = BatchEnvRL(n_envs=args.batch_size, seed=seed, n_nodes=n_nodes, noise_on=args.noise_on)

    run_episode.train()
    sim_av_rew = 0
    sim_av_rwds = 0
    sim_av_pens = 0
    sim_av_loss = 0

    for sim in range(n_sims):

        inst_features, max_travel_times = env.get_our_features(feature_list)
        # inst_features, max_travel_times = env.get_our_features_old()

        inst_features = torch.from_numpy(inst_features.astype(np.float32).copy()).to(args.device)
        max_travel_times = torch.from_numpy(max_travel_times.copy()).to(args.device)
        inst_features_scaled = data_scaler(inst_features, args.feature_list, n_nodes)

        actions, logits, entropy, seq_mask = run_episode(env, inst_features,
                                                              inst_features_scaled, max_travel_times,
                                                              'stochastic')
        rwds, pens = env.check_solution(actions)
#         print (actions)
#
        rewards = rwds + pens
        rewards = torch.from_numpy(rewards).to(args.device)

        if sim == 0:
            exp_mvg_avg = rewards.mean()
        else:
            exp_mvg_avg = (exp_mvg_avg * args.beta) + ((1. - args.beta) * rewards.mean())

        loss = 0

        av_rew = rewards.mean()
        av_rwds = rwds.mean()
        av_pens = pens.mean()
        #min_rew = rewards.min()
        #max_rew = rewards.max()
        advantage = (rewards - exp_mvg_avg) #advantage

        output = -1 * advantage * logits + args.gamma * entropy
        loss = output[seq_mask].sum() / torch.sum(seq_mask)

        loss.backward()
        if (sim+1) % args.accumulation_steps == 0:  # Wait for several backward steps
            nn.utils.clip_grad_norm_(run_episode.parameters(), args.max_grad_norm)
            opt.step()
            opt.zero_grad()
            scheduler.step()


        sim_av_rwds += av_rwds.item()
        sim_av_pens += av_pens.item()

        sim_av_rew += av_rew.item()
        sim_av_loss += loss.item()

        exp_mvg_avg = exp_mvg_avg.detach()
        env.reset()

    return sim_av_rew/n_sims, sim_av_loss/n_sims, sim_av_rwds/n_sims, sim_av_pens/n_sims

def predict(env, inst_features, max_travel_times, run_episode, ds, args):
    # evaluate
    feature_list = args.feature_list
    with torch.no_grad():  # operations inside don't track history
        run_episode.eval()

        inst_features = torch.from_numpy(inst_features.astype(np.float32).copy()).to(args.device)
        max_travel_times = torch.from_numpy(max_travel_times.copy()).to(args.device)
        inst_features_scaled = ds.data_scaler(inst_features, args.feature_list, env.n_nodes)

        actions, logits, entropy, seq_mask = run_episode(env, inst_features,
                                                              inst_features_scaled, max_travel_times,
                                                              'greedy')
        rwds, pens = env.check_solution(actions)
    return actions, rwds, pens


def active_search_train_model(run_episode, env, ds, opt, args):

    opt.zero_grad()
    n_sims_as = args.n_sims_as
    feature_list = args.feature_list

    run_episode.train()
    sim_av_rew = 0
    sim_av_rwds = 0
    sim_av_pens = 0
    sim_av_loss = 0

    for sim in tqdm(range(n_sims_as), leave=False):
        opt.zero_grad()

        inst_features, max_travel_times = env.get_our_features(feature_list)
        # inst_features, max_travel_times = env.get_our_features_old()

        inst_features = torch.from_numpy(inst_features.astype(np.float32).copy()).to(args.device)
        max_travel_times = torch.from_numpy(max_travel_times.copy()).to(args.device)
        inst_features_scaled = ds.data_scaler(inst_features, args.feature_list, env.n_nodes)

        actions, logits, entropy, seq_mask = run_episode(env, inst_features,
                                                              inst_features_scaled, max_travel_times,
                                                              'stochastic')
        rwds, pens = env.check_solution(actions)

        rewards = rwds + pens
        rewards = torch.from_numpy(rewards).to(args.device)

        rewards[rewards.le(-1)]= -1

        if sim == 0:
            exp_mvg_avg = rewards.median()
        else:
            exp_mvg_avg = (exp_mvg_avg * args.beta) + ((1. - args.beta) * rewards.median())

        loss = 0

        av_rew = rewards.mean()
        av_rwds = rwds.mean()
        av_pens = pens.mean()
        min_rew = rewards.min()
        max_rew = rewards.max()

        #print(min_rew.item(), max_rew.item(), av_rew.item(), exp_mvg_avg.item())
        advantage = (rewards - exp_mvg_avg) #advantage

        output = -1 * advantage * logits + args.gamma * entropy
        loss = output[seq_mask].sum() / torch.sum(seq_mask)

        loss.backward()
        nn.utils.clip_grad_norm_(run_episode.parameters(), args.max_grad_norm)
        opt.step()

        sim_av_rwds += av_rwds.item()
        sim_av_pens += av_pens.item()

        sim_av_rew += av_rew.item()
        sim_av_loss += loss.item()

        exp_mvg_avg = exp_mvg_avg.detach()
        env.reset()

    return sim_av_rew/n_sims_as, sim_av_rwds/n_sims_as, sim_av_pens/n_sims_as


def as_inference(env, run_env_episode, ds, args):

    opt = optim.Adam(run_env_episode.parameters(), lr=args.lr_as)

    avreward, avg_rwds, avg_pen = active_search_train_model(run_env_episode, env, ds, opt, args)

    return run_env_episode


def run_validation(run_episode, range_lb, range_ub, ds, args, which_set='valid'):
    feature_list = args.feature_list
    insts = [f"{i:04}" for i in range(range_lb, range_ub+1)]

    reward_val =  0
    penalty_val = 0
    assert which_set in ['test', 'valid'], 'choose which_set between test and valid'
    if which_set == 'valid':
        seed = 12345
    elif which_set == 'test':
        seed = 19120623



    for inst in tqdm(insts):
        x_path=f'../data/{which_set}/instances/instance{inst}.csv'
        adj_path=f'../data/{which_set}/adjs/adj-instance{inst}.csv'

        env = BatchEnvRL(n_envs=1, from_file=True,
                                seed=seed,
                                x_path=x_path,
                                adj_path=adj_path,
                                noise_on=True)


        inst_features, max_travel_times = env.get_our_features(feature_list)

        _, rwds, pens = predict(env, inst_features, max_travel_times, run_episode, ds, args)
        reward_val += rwds[0][0]
        penalty_val += pens[0][0]
        env.reset()
    return reward_val/len(insts), penalty_val/len(insts)

def average_snapshots(swa_model, model, list_of_snapshots_epochs, args):    
    
    for snapshot_epoch in list_of_snapshots_epochs:
        checkpoint = torch.load('{path}/model_{agent_name}_noise_{noise}_{notebook_name}_epoch_{epoch}_r0.pkl'.format(path=args.save_weights_dir, 
                                   agent_name=args.agent_name,
                                   noise=str(int(args.noise_on)),
                                   notebook_name=args.nb_name,
                                   epoch=snapshot_epoch))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        swa_model.update_parameters(model)
        
    swa_model.eval()

    return swa_model


def create_submission(run_episode, ds, args, n_tours=1, with_as=False, which_set=None, with_wa=False):

    feature_list = args.feature_list
    #insts = [f"{i:04}" for i in range(1, 11)]+[f"{i:04}" for i in range(251, 261)]+[f"{i:04}" for i in range(501, 511)]
    insts = [f"{i:04}" for i in range(1, 1001)]
    assert which_set in ['test', 'valid'], 'choose which_set between test and valid'

    if which_set == 'valid':
        submission_filepath = '../baseline_rl/example_output_rl_validation.json'
    elif which_set == 'test':
        submission_filepath = '../baseline_rl/example_output_rl.json'

    f = open(submission_filepath)
    submission = json.load(f)

    submission = {}
    reward_val =  0
    penalty_val = 0
    new_sub = submission
    if which_set == 'valid':
        seed = 12345
    elif which_set == 'test':
        seed = 19120623

    count = 0
    for inst in tqdm(insts):

        instance_name = 'instance' +inst
        x_path=f'../data/{which_set}/instances/{instance_name}.csv'
        adj_path=f'../data/{which_set}/adjs/adj-{instance_name}.csv'

        if with_as:
            env_train = BatchEnvRL(n_envs=args.batch_size, from_file=True,
                                            seed=seed+128, # to do active search with a different seed
                                            x_path=x_path,
                                            adj_path=adj_path,
                                            noise_on=True)
            run_env_episode = copy.deepcopy(run_episode)
            run_env_episode = as_inference(env_train, run_env_episode, ds, args)
        else:
            run_env_episode = copy.deepcopy(run_episode)


        env = BatchEnvRL(n_envs=1, from_file=True,
                                seed=seed,
                                x_path=x_path,
                                adj_path=adj_path,
                                noise_on=True)

        sims = {'tours':{}}

        for tour in range(1, n_tours+1):

            tour_name = 'tour'+f"{tour:03}"

            inst_features, max_travel_times = env.get_our_features(feature_list)

            assert tour_name == env.get_sim_name()[0], f'submission {tour_name} in {instance_name} is in the wrong order.'
            actions, rwds, pens = predict(env, inst_features, max_travel_times, run_env_episode, ds, args)

            sims['tours'][tour_name] = actions.cpu().tolist()[0]
            count +=1
            reward_val += rwds[0][0]
            penalty_val += pens[0][0]
            env.reset()

            #print(rwds[0][0], pens[0][0])

        submission[instance_name] = sims
        submission[instance_name]['seed'] = seed
        submission[instance_name]['nodes'] = max_travel_times.shape[1]


    print(which_set, " - reward: %2.3f, penalty: %2.3f" %(reward_val / count, penalty_val / count))

    if with_as:
        method = 'as'
    elif with_wa:
        method = 'wa'
    else:
        method = 'gr'

    file_path = '{path}/submission_{agent_name}_noise_{noise}_{notebook_name}_{method}_{which_set}.json'.format(path=args.save_sub,
                                           agent_name=args.agent_name,
                                           noise=str(int(args.noise_on)),
                                           notebook_name=args.nb_name,
                                           method = method,
                                           which_set=which_set)

    with open(file_path, 'w') as outfile:
        json.dump(new_sub, outfile)
    return reward_val / count, penalty_val / count
