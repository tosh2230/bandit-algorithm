from random import random


class BernoulliArm:
    def __init__(self, p) -> None:
        self.p = p

    def draw(self):
        if random() > self.p:
            return 0.0
        else:
            return 1.0
