from random import random, randrange

from utility import ind_max


class EpsilonGreedy:
    def __init__(self, epsilon: float, counts: list[int], values: list[float]):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms: int) -> None:
        self.counts: list[int] = [0] * n_arms
        self.values: list[float] = [0.0] * n_arms

    def select_arm(self) -> float:
        if random() > self.epsilon:
            return ind_max(self.values)
        else:
            return randrange(len(self.values))

    def update(self, chosen_arm: int, reward: float) -> None:
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n: int = self.counts[chosen_arm]

        value: float = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
