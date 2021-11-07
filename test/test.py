from reporter import Reporter
import numpy as np
import random

matrix = np.array([[float('inf'),3,2,3],[3,float('inf'),2,1],[1,3,float('inf'),2],[2,3,2,float('inf')]])
num_cities = 29
k = 4
population = np.array([[0,2,3,1], [2,3,1,0], [3,2,1,0], [2,1,3,0], [1,3,0,2]])
sol = np.array([25,27,28,22,21,20,16,17,18,14,11,10,9,5,0,1,4,7,3,2,6,8,12,13,15,23,24,26,19])



def objf(candidate):
    sum = 0
    test_file = open("./tour29.csv")
    weights = np.loadtxt(test_file, delimiter=",")
    for i in range(num_cities - 1):
        city_1 = candidate[i]
        city_2 = candidate[i + 1]
        weight = weights[city_1, city_2]
        sum += weight
    last_and_first_weight = weights[candidate[num_cities - 1], candidate[0]]
    sum += last_and_first_weight
    return sum

print(objf(sol))

def objfpop(candidates):
    array = np.zeros(k)
    for i in range(k):
        array[i] = objf(candidates[i])
    return array




def selection():
    selected = np.zeros((2, num_cities))
    for ii in range(2):
        ri = random.choices(range(5), k=k)  # saving k indexes from the population
        min = np.argmin(objfpop(population[ri, :]))  # find best index
        selected[ii, :] = population[ri[min], :]
    return selected

#print(selection())