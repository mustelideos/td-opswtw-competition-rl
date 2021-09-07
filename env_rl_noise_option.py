from env_rl import EnvRL
import numpy as np

class EnvRLNoiseOption(EnvRL):
    def __init__(self, n_nodes=None, seed=None, from_file=False, x_path=None, adj_path=None, verbose=False,
                 adaptive=True, noise_on=True):
        super().__init__(n_nodes, seed, from_file, x_path, adj_path, verbose, adaptive)
        self.noise_on = noise_on

    def step(self, node):

        if len(self.tour) >= self.n_nodes + 1:
            return None
        assert node <= self.n_nodes, f'node {node} does not exist for instance of size {self.n_nodes}'

        previous_tour_time = self.tour_time
        time = self.adj[self.current_node - 1, node - 1]
        if self.noise_on:
            noise = np.random.randint(1, 101, size=1)[0] / 100
        else:
            noise = 1.0
        self.tour_time += np.round(noise * time, 2)
        self._get_rewards(node)
        self.time_t = self.tour_time - previous_tour_time
        self.current_node = node

        return self.tour_time, self.time_t, self.rwd_t, self.pen_t, self.feas, self.violation_t, self.return_to_depot



if __name__ == '__main__':

    print('no noise')
    env = EnvRLNoiseOption(5, seed=12345, noise_on=False)
    print('name', env.name)
    env.step(2)
    env.step(4)
    env.step(5)
    env.step(1)
    env.step(3)
    print('tour', env.tour)
    print('tour time', env.tour_time)
    print(50*'-')
    env.reset()
    print('name', env.name)
    env.step(2)
    env.step(4)
    env.step(5)
    env.step(1)
    env.step(3)
    print('tour', env.tour)
    print('tour time', env.tour_time)

    print(50*'-')
    print('with noise')
    print(50*'-')
    env = EnvRLNoiseOption(5, seed=12345)
    print('name', env.name)
    env.step(2)
    env.step(4)
    env.step(5)
    env.step(1)
    env.step(3)
    print('tour', env.tour)
    print('tour time', env.tour_time)
    print(50*'-')
    env.reset()
    print('name', env.name)
    env.step(2)
    env.step(4)
    env.step(5)
    env.step(1)
    env.step(3)
    print('tour', env.tour)
    print('tour time', env.tour_time)
