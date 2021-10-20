from reporter import Reporter
import numpy as np

class Individual:

	def __init__(self):
		# self.solution = np.array... 1D
		pass


# Modify the class name to match your student number. Evolutionary algorithm
class r0123456:

	# In constructor you should get mutation rate, number of iterations, population size, offspring size
	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		# self.weights = None numpy 2D array
		# number of iterations
		# mutation rate
	    # population size=lambda
		# offspring size= mu
		# self.population = self.initialize

	def initialize(self):
		"""
		Return population
		:return:
		"""
		pass

	def selection(self):
		"""
		Return one parent. Perform k-tournament selection
		:return:
		"""
		pass

	def crossover(self, individual1, individual2):
		"""
		Return two offsprings
		:param individual1:
		:param individual2:
		:return: two offsprings
		"""
		pass

	def mutation(self, population):
		"""
		Population is here parent population + all offsprings.
		:param population:
		:return: mutated population
		"""
		pass

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.

		while( yourConvergenceTestsHere ):

			# Your code here.
			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break

		# Your code here.
		return 0

	def selection(self):
		pass


if __name__ == "__main__0":
	pass
