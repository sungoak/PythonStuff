from numpy import random
class Cheese(object):
    def __init__(self, num_holes=0):
        "defaults to a solid cheese"
        self.number_of_holes = num_holes

    @classmethod
    def random(cls):
        print random(100),cls(random(100))
        return cls(random(100))

    @classmethod
    def slightly_holey(cls, num):
        print random(33),cls(random(33))
        return cls(random(33))

    @classmethod
    def very_holey(cls, num):
        print random(66,100),cls(random(66,100))
        return cls(random(66, 100))
