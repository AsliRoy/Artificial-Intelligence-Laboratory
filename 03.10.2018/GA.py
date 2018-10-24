import random
import numpy as np
def init_population(size):
	population = []
	for p in range(size):
		single = []
		for q in range(45):
			single.append(random.getrandbits(1))
		population.append(single)
	return population

def bin_to_float(neuron):
	out = 0
	for bit in neuron[1:]:
		out = (out << 1) | bit
	out = out / 4
	if neuron[0]:
		return -1*out
	return out

def calculate_weights(weights):
	single = []
	for t in range(9):
		value = bin_to_float(weights[(5*t):(5*(t+1))])
		single.append(value)
	return single

def calculate_out(weights, in1, in2):
	neuron_1_1 = np.array(weights[0:3])
	neuron_1_2 = np.array(weights[3:6])
	neuron_2_1 = np.array(weights[6:])
	n_input = np.array([in1, in2, 1])
	hidden_layer = np.array([ threshold(np.dot(n_input, neuron_1_1)), threshold(np.dot(n_input, neuron_1_2)), 1 ])
	output = threshold(np.dot(hidden_layer, neuron_2_1))
	return output

def threshold(arg):
	if arg>=0:
		return 1
	return 0

def fitness_function(bin_arg, nobin=False, debug=False):
	expected = [0, 1, 1, 0]
	input_list = [(0, 0), (0, 1), (1,0), (1,1)]
	sum_error = 0
	if nobin:
		single_weights = bin_arg
	else:
		single_weights = calculate_weights(bin_arg)
	for t in range(4):
		if debug:
			print("expected: "+str(expected[t]))
			print("input: "+str(input_list[t][0])+"\t"+str(input_list[t][1]))
			t_out = calculate_out(single_weights, input_list[t][0], input_list[t][1])
			print("output: "+str(t_out))
			sum_error += abs(expected[t] - t_out)
		else:
			sum_error += abs(expected[t] - calculate_out(single_weights, input_list[t][0], input_list[t][1]))
	return (4-sum_error)

# change one bit of phenotype
def mutation(single):
	bit = random.choice(range(45))
	single[bit] = int(not single[bit])
	return single

def single_crossover(wife, husband):
	split = random.choice(range(45))
	daughter = wife[:split] + husband[split:]
	son = husband[:split] + wife[split:]
	return daughter, son

def population_crossover(population):
	old_size = len(population)
	surviving_parents = int(0.33*old_size)
	new_generation = []
	while len(population) >= 2 and len(population) >= surviving_parents:
		wife = population.pop(random.randrange(len(population)))
		husband = population.pop(random.randrange(len(population)))
		daughter, son = single_crossover(wife, husband)
		new_generation.append(mutation(daughter))
		new_generation.append(mutation(son))
	return new_generation

# operator of selection shrinks population size
def selection(population, p_size):
	ranking = [fitness_function(t) for t in population]
	sum_rank = float(sum(ranking))
	percentage_rank = [ p / sum_rank for p in ranking]
	thresholds = [ sum( percentage_rank[:(r+1)] ) for r in range(len(percentage_rank)) ]
	new_population = []
	for q in range(p_size):
		rand = random.uniform(0, 1)
		for(index, single) in enumerate(population):
			if rand<=thresholds[index]:
				new_population.append(single)
				break
	return new_population


if __name__ == '__main__':
	
	population = init_population(10)
	iterator = 0
	while True:
		#print("population size: "+str(len(population)))
		parents = selection(population, int(len(population)*0.6))
		for p in parents:
			try:
				population.remove(p)
			except:
				pass
		children = population_crossover(parents)
		population += children
		ranking = [fitness_function(s) for s in population]

		if 4 in ranking:
			print("iterations: "+str(iterator))
			winners = [q for q in population if fitness_function(q)==4]
			print("winners: "+str(len(winners)))
			print(winners)
			winner_weights = []
			for r in winners:
				temp = calculate_weights(r) 
				winner_weights.append(temp)
				print(temp)
				print(fitness_function(r, debug=True))
			break
		else:
			iterator += 1