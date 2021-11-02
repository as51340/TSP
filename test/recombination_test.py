import numpy as np
import random

p1 = np.array([1,2,3,4,5,6,7,8,9])
p2 = np.array([9,3,7,8,2,6,5,1,4])


def parse(input, start, stop):
    if start > stop:
        return np.concatenate([input[start:], input[:stop]])
    return input[start:stop]

def recombination(p1, p2):
    # i = random.randint(0, 4 - 1)
    # j = random.randint(0, 4 - 1)
    i = 7
    j = 3

    slice_P1 = parse(p1, i, j)
    candidate_P2 = np.roll(p2, -(j))
    no_dup = np.setdiff1d(candidate_P2, slice_P1, assume_unique=True)
    print(slice_P1)
    print(candidate_P2)
    print(no_dup)
    not_ordered_sol = np.concatenate([slice_P1, no_dup])
    sol = np.roll(not_ordered_sol, i)
    print(not_ordered_sol)
    print(sol)
    return slice_P1

recombination(p1,p2)
