from copy import deepcopy
from random import shuffle, seed

from algorithms.epsilon_greedy import EpsilonGreedy
from arms.bernoulli import BernoulliArm
from utility import ind_max


def test_algorithm(algo, arms, num_sims, horizon) -> list:
    init_value = [0.0 for _ in range(num_sims * horizon)]
    chosen_arms = deepcopy(init_value)
    rewards = deepcopy(init_value)
    cumulative_rewards = deepcopy(init_value)
    sim_nums = deepcopy(init_value)
    times = deepcopy(init_value)

    for sim in range(num_sims):
        sim += 1
        algo.initialize(len(arms))

        for t in range(horizon):
            t += 1
            index = (sim - 1) * horizon + t - 1
            sim_nums[index] = sim
            times[index] = t

            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm
            reward = arms[chosen_arms[index]].draw()
            rewards[index] = reward

            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algo.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


if __name__ == "__main__":
    seed(1)
    means = [0.1, 0.1, 0.1, 0.1, 0.9]
    n_arms = len(means)
    shuffle(means)
    arms = [BernoulliArm(p=m) for m in means]
    print(f"Best arm is {str(ind_max(means))}")

    epsilon_attempts = [0.1, 0.2, 0.3, 0.4, 0.5]
    for idx, epsilon in enumerate(epsilon_attempts):
        algo = EpsilonGreedy(epsilon=epsilon, counts=[], values=[])
        algo.initialize(n_arms=n_arms)
        results = test_algorithm(algo=algo, arms=arms, num_sims=5000, horizon=250)

        with open(f"standard_results_{idx + 1}.tsv", "w") as f:
            for i in range(len(results[0])):
                f.write(
                    "\t".join([str(results[j][i]) for j in range(len(results))]) + "\n"
                )
