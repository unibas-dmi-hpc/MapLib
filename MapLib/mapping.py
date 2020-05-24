from collections import deque
from heapq import heappop, heappush
from itertools import product
from random import sample, randint, random, uniform
import copy
import nxmetis
import networkx as nx
import sys
import math

from .vector import Vector
from .routing import manhattan_distance

import cProfile
def profile(func):
	""" Profiling function"""
	def profiled_func(*args, **kwargs):
		profile = cProfile.Profile()
		try:
			profile.enable()
			result = func(*args, **kwargs)
			profile.disable()
			return result
		finally:
			profile.print_stats()
	return profiled_func


class Mapping():
	"""Base class for all mapping strategies."""

	@profile
	def __init__(self, process_graph, topology, **kwargs):
		self.routed = False
		self.process_graph = process_graph
		self._topology = topology
		for key, value in kwargs.items():
			setattr(self, key, value)
		self.mapping = self.map()

	@property
	def rev_mapping(self, mapping=None):
		"""
		Returns the reversed mapping
		:param mapping: Mapping
		:return: Reversed mapping
		"""
		items = mapping.items() if mapping else self.mapping.items()
		reverse = {}
		for proc, cord in items:
			try:
				reverse[cord].append(proc)
			except KeyError:
				reverse[cord] = [proc]
		return reverse

	@property
	def name(self):
		"""
		Returns the name of the mapping strategy.
		:return: Mapping name
		"""
		return self._name

	@property
	def topology(self):
		"""
		Returns the topology for mapping.
		:return: Topology
		"""
		return self._topology

	def to_file(self, f):
		"""
		Writes mapping to file
		:param f: Filename
		"""
		f.write(self.__str__())

	def cardinality(self, mapping):
		"""
		Returns the cardinality of a mapping
		:param mapping: Mapping
		:return: Cardinality
		"""
		_sum = 0
		for u, v in self.process_graph.edges():
			if self.topology.are_neighbors(mapping[u], mapping[v]):
				_sum += 1
		return _sum

	def __str__(self):
		"""
		Returns the string representation of a mapping
		:return: Output mapping representation
		"""
		s = self.__class__.__name__.lower() + '\n'
		s += 'x_coord y_coord z_coord number_of_processes process_id(s)\n'
		for (x, y, z), proc_list in sorted(self.rev_mapping.items()):
			s += '%d %d %d %d ' % (x, y, z, len(proc_list))
			s += ' '.join([str(i) for i in proc_list]) + '\n'
		return s


class UDFS(Mapping):
	"""Class for utilization-based depth-first mapping"""
	"""Slightly changed: instead of choosing just next processor,"""
	"""we look for the closest one to already mapped (if task communicate)"""
	def map(self):
		sys.setrecursionlimit(10000)
		task_graph = self.process_graph
		processor_graph = nx.Graph()
		for elem in range(len(self.topology)):
			x,y,z = self.topology.to_xyz(elem)
			for neighbour in self.topology.neighbours(x,y,z):
				processor_graph.add_edge(elem,self.topology.to_idx(neighbour))
		mapping = {}

		def UDFS_algorithm():
			#queue_of_heaviest_task = [i[0] for i in sorted(task_graph.degree(None,'weight').items(), key=lambda item:item[1], reverse=True)]
			queue_of_heaviest_task = [i[0] for i in sorted(task_graph.degree(None,'weight'), key=lambda item:item[1], reverse=True)]
			processor_set = {i for i in range(len(processor_graph))}
			start_node = len(processor_set)//2
			start_task = queue_of_heaviest_task.pop(0)
			mapping[start_task] = self.topology.to_xyz(start_node)
			processor_set.remove(start_node)
			set_mapped = set()
			set_mapped.add(start_task)

			def map_next(i,p):
				while not set(task_graph.neighbors(i)).issubset(set_mapped):
					queue = []
					for neighbour in task_graph.neighbors(i):
						if neighbour not in set_mapped:
							queue.append((neighbour,task_graph.get_edge_data(i,neighbour)['weight']))
					queue = [elem[0] for elem in sorted(queue,key=lambda item:item[1],reverse=True)]
					near_node = [edge[0] for edge in sorted(nx.single_source_shortest_path_length(processor_graph,self.topology.to_idx(mapping[i])).items(),key=lambda item:item[1]) if edge[1]!=0]
					for node in near_node:
						if node in processor_set:
							break
					k = queue.pop(0)
					mapping[k] = self.topology.to_xyz(node)
					processor_set.remove(node)
					set_mapped.add(k)
					p = map_next(k,node)
				return p

			map_next(start_task,start_node)

		UDFS_algorithm()
		return mapping


class Greedy_All_C(Mapping):
	"""Class for GreedyAllC mapping from Glantz and Meyerhenke article"""
	def map(self):
		task_graph = self.process_graph
		processor_graph = nx.Graph()
		for elem in range(len(self.topology)):
			x,y,z = self.topology.to_xyz(elem)
			for neighbour in self.topology.neighbours(x,y,z):
				processor_graph.add_edge(elem,self.topology.to_idx(neighbour))
		mapping = {}

		flag = False #false if without partition, true - otherwise

		def create_partition_graph(tgraph,pgraph):
			partition = nxmetis.partition(tgraph,len(pgraph),None,None,'weight',None,None,None,True)
			partition_graph = nx.Graph()
			partition_graph.add_nodes_from(pgraph.nodes())
			for i in range(len(partition[1])):
				for j in range(len(partition[1])):
					if j != i:
						val = 0
						for elem_from_i in partition[1][i]:
							for elem_from_j in partition[1][j]:
								if tgraph.has_edge(elem_from_i,elem_from_j):
									val += tgraph.get_edge_data(elem_from_i,elem_from_j)['weight']
						if val > 0:
							partition_graph.add_edge(i,j,weight=val)
			return partition,partition_graph


		def compute_tP():
			t = {}
			tP_min = {}
			for u_p in processor_graph.nodes():
				t[u_p] = []
				u_p_xyz = self.topology.to_xyz(u_p)
				for v_p in processor_graph.nodes():
					v_p_xyz = self.topology.to_xyz(v_p)
					if u_p == v_p:
						distance = 0
					elif self.topology.are_neighbors(u_p_xyz,v_p_xyz):
						distance = 1
					else:
						distance = manhattan_distance(u_p_xyz,v_p_xyz)
					t[u_p].append(distance)
				tP_min[u_p] = sum(t[u_p])
			return tP_min, t

		if flag:
			partition, partition_graph = create_partition_graph(task_graph,processor_graph)
		else:
			partition_graph = task_graph


		return_mapping = {}

		sum_min, tP = compute_tP()
		#v_0_c = max(partition_graph.degree(None,'weight').items(),key=lambda item:item[1])[0]
		v_0_c = max(partition_graph.degree(None,'weight'),key=lambda item:item[1])[0]
		v_0_p = min(sum_min.items(),key=lambda item:item[1])[0]

		sum_c = [0 for i in range(len(partition_graph))]
		sum_p = [1 for i in range(len(partition_graph))]

		for i in range(len(partition_graph)):
			sum_c[v_0_c] = -1
			sum_p[v_0_p] = sys.maxsize
			mapping[v_0_c] = self.topology.to_xyz(v_0_p)
			for w in partition_graph[v_0_c]:
				if sum_c[w] >= 0:
					sum_c[w] = sum_c[w] + partition_graph.get_edge_data(v_0_c,w)['weight']

			v_0_c = sum_c.index(max(sum_c))
			for j in range(len(processor_graph)):
				if sum_p[j] < sys.maxsize:
					sum_p[j] = 0
					for w in partition_graph[v_0_c]:
						if sum_c[w] < 0:
							sum_p[j] = sum_p[j] + partition_graph.get_edge_data(v_0_c,w)['weight'] * tP[j][self.topology.to_idx(mapping[w])]
			v_0_p = sum_p.index(min(sum_p))

		if flag:
			for i in range(len(mapping)):
				for elem in partition[1][i]:
					return_mapping[elem] = mapping[i]
			return return_mapping
		else:
			return mapping




class Greedy_Graph_Embedding(Mapping):
	"""Class for greedy graph embedding mapping"""
	def map(self):
		task_graph = self.process_graph 
		processor_graph = nx.Graph() #create a topology graph using info about each node`s neighbours
		for elem in range(len(self.topology)):
			x,y,z = self.topology.to_xyz(elem)
			for neighbour in self.topology.neighbours(x,y,z):
				processor_graph.add_edge(elem,self.topology.to_idx(neighbour)) #if two nodes are neighbours, we add an edge between them to our topo_graph
		dijkstra_path = dict(nx.all_pairs_dijkstra_path_length(processor_graph))

		flag = False #false if without partition, true - with

		def find_new_start(C,s):#we are lookong for the nearest node to s that is free for allocating, i.e. has C(s)>0
			left,right = 0,0
			while True:
				if C[s-left]==1:
					return s-left
				elif C[s+right]==1:
					return s+right
				if s-left>0:	
					left += 1
				if s+right<len(C)-1:
					right += 1

		def create_partition_graph(tgraph,pgraph):#we create a partition of a given app_graph to len(graph)-chunks, 
												  #several tasks that communicate tightly are combined into one block, an edge between each block has weight equal to sum of all edges between all tasks in two different blocks
												  #
			partition = nxmetis.partition(tgraph,len(pgraph),None,None,'weight',None,None,None,True)
			partition_graph = nx.Graph()
			partition_graph.add_nodes_from(pgraph.nodes())
			for i in range(len(partition[1])):
				for j in range(len(partition[1])):
					if j != i:
						val = 0
						for elem_from_i in partition[1][i]:
							for elem_from_j in partition[1][j]:
								if tgraph.has_edge(elem_from_i,elem_from_j):
									val += tgraph.get_edge_data(elem_from_i,elem_from_j)['weight']
						if val>0:
							partition_graph.add_edge(i,j,weight=val)
			return partition, partition_graph

		def dijkstra_closest_vertex(C,start): #here we are looking for the nearest node to our start position
											  #that has C[s]>0
			min_distance = float("inf")
			for elem in dijkstra_path[start]:
				if elem != start and C[elem]>0:
					if dijkstra_path[start][elem] < min_distance:
						min_distance,min_index = dijkstra_path[start][elem],elem
			return min_index
			
		def greedy_graph_embedding_algo(tgraph,pgraph):
			mapping = {}
			task_set = set(tgraph.nodes())
			C = [1 for i in range(len(pgraph))]
			queue = []
			start_node = list(pgraph.nodes())[len(pgraph.nodes())//2] #one from the center ID`s
			#here we sort all vertices according the weights of their out edges
			queue_of_heaviest_in_S = [i[0] for i in sorted(tgraph.degree(None,'weight'),key=lambda item:item[1],reverse=True)]
			#queue_of_heaviest_in_S = [i[0] for i in sorted(tgraph.degree(None,'weight'),key=lambda item:item[1],reverse=True)]
			while len(task_set)!=0:
				vertex_m = queue_of_heaviest_in_S.pop(0)
				if C[start_node] == 0:
					start_node = find_new_start(C,start_node)
				mapping[vertex_m] = self.topology.to_xyz(start_node)
				task_set.remove(vertex_m)
				C[start_node] = 0
				for u in tgraph[vertex_m]:
					if u in task_set:
						queue.append((vertex_m,u,tgraph.get_edge_data(vertex_m,u)))
				#here we sort the queue according the weights of each edge		
				queue = sorted(queue,key=lambda x:x[2]['weight'],reverse=True)
				while len(queue)!=0:
					heaviest_edge_in_Q = queue.pop(0)
					if C[start_node] == 0:
						start_node = dijkstra_closest_vertex(C,start_node)
					_,vertex_m = heaviest_edge_in_Q[0],heaviest_edge_in_Q[1]
					mapping[vertex_m] = self.topology.to_xyz(start_node)
					task_set.remove(vertex_m)
					queue_of_heaviest_in_S.remove(vertex_m)
					C[start_node] = 0
					#here we have to check whether our u in task_set or not
					#if not, there can be a situation, when an edge with its neighbour is already added into queue,
					#hence we have to delete it from the queue in order not consider it in the future (because we already have allocated these two nodes onto topology nodes)
					for u in tgraph[vertex_m]:
						if u in task_set:
							queue.append((vertex_m,u,tgraph.get_edge_data(vertex_m,u)))
						elif (vertex_m,u,tgraph.get_edge_data(vertex_m,u)) in queue or (u,vertex_m,tgraph.get_edge_data(vertex_m,u)) in queue:
							try:
								queue.remove((vertex_m,u,tgraph.get_edge_data(vertex_m,u)))
							except ValueError:
								queue.remove((u,vertex_m,tgraph.get_edge_data(vertex_m,u)))
					queue = sorted(queue,key=lambda x:x[2]['weight'],reverse=True)
			return mapping

		if flag:
			partition, partition_graph = create_partition_graph(task_graph,processor_graph)
			maps = greedy_graph_embedding_algo(partition_graph,processor_graph)
			return_mapping = {}
			for i in range(len(maps)):
				for elem in partition[1][i]:
					return_mapping[elem]=maps[i]
			return return_mapping
		else:
			maps = greedy_graph_embedding_algo(task_graph,processor_graph)
			return maps
		#returned mapping (maps) will consist of tuples: id of block and coordinates of topology nodes
		#however each block can consist of several tasks ids, respectively
		#so, now we create a new dictionar: id of task, coordinates of topo node
										  			


class Recursive_bipartitioning(Mapping):
	"""Class for the recursive bipartitioning mapping"""
	def map(self):
		mapping={}
		task_graph = self.process_graph
		processors_set = {i for i in range(len(self.topology))}
		processor_graph = nx.Graph()
		for elem in range(len(self.topology)):
			x,y,z = self.topology.to_xyz(elem)
			for neighbour in self.topology.neighbours(x,y,z):
				processor_graph.add_edge(elem,self.topology.to_idx(neighbour))
		len_graph_part = 0


		def node_partition(set_of_nodes):#if our topo is represented as a set 
			if isinstance(set_of_nodes,set):
				set_of_nodes = list(set_of_nodes)
			global len_graph_part
			half = len_graph_part
			return set_of_nodes[:half], set_of_nodes[half:]

		def graph_node_partition(topology_graph):#proper case, if our topo is a graph
			partition = nxmetis.partition(topology_graph,2,None,None,None,None,None,None,True)
			# LLP: checking if the partitions have the same size
			# if that is not the case, we move items from the biggest to the smallest until equilibrium is met
			if(len(partition[1][0]) != len(partition[1][1])):
				left_is_smallest = len(partition[1][0]) < len(partition[1][1]) # Gets who is the smallest one
				small, big = ((partition[1][1], partition[1][0]), (partition[1][0], partition[1][1]))[left_is_smallest] # calls the smallest as "small"
				while len(small) < len(big):
					small.append(big.pop(0))
			left_graph = nx.Graph()
			left_graph.add_nodes_from(partition[1][0])
			for u in partition[1][0]:
				for v in partition[1][0]:
					if processor_graph.has_edge(u,v):
						left_graph.add_edge(u,v)
			right_graph = nx.Graph()
			right_graph.add_nodes_from(partition[1][1])
			for u in partition[1][1]:
				for v in partition[1][1]:
					if processor_graph.has_edge(u,v):
						right_graph.add_edge(u,v)
			return left_graph,right_graph

		def graph_partition(communication_graph):
			partition = nxmetis.partition(communication_graph,2,None,None,'weight',None,None,None,True)
			# LLP: checking if the partitions have the same size
			# if that is not the case, we move items from the biggest to the smallest until equilibrium is met
			if(len(partition[1][0]) != len(partition[1][1])):
				left_is_smallest = len(partition[1][0]) < len(partition[1][1]) # Gets who is the smallest one
				small, big = ((partition[1][1], partition[1][0]), (partition[1][0], partition[1][1]))[left_is_smallest] # calls the smallest as "small"
				while len(small) < len(big):
					small.append(big.pop(0))
			left_graph=nx.Graph()
			left_graph.add_nodes_from(partition[1][0])
			for u in partition[1][0]:
				for v in partition[1][0]:
					if task_graph.has_edge(u,v):
						val = task_graph.get_edge_data(u,v)['weight']
						left_graph.add_edge(u,v,weight=val)
			right_graph=nx.Graph()
			right_graph.add_nodes_from(partition[1][1])
			for u in partition[1][1]:
				for v in partition[1][1]:
					if task_graph.has_edge(u,v):
						val = task_graph.get_edge_data(u,v)['weight']
						right_graph.add_edge(u,v,weight=val)
			global len_graph_part
			len_graph_part = len(left_graph)
			return left_graph,right_graph
			
		def bipartitioning_mapping(communication_graph,set_of_nodes):
			if (len(set_of_nodes)==1):
				if len(task_graph) >= 256:
					mapping[communication_graph.nodes()[0]]=self.topology.to_xyz(set_of_nodes[0]) #if our topology is just a SET = {0,..,63}, for example, of allocated to app nodes
				else:
					mapping[list(communication_graph.nodes)[0]]=self.topology.to_xyz(list(set_of_nodes.nodes)[0]) #if our topology is a graph with edges between each node
				return
			G_1, G_2 = graph_partition(communication_graph)
			if len(task_graph) < 256:
				N_1,N_2 = graph_node_partition(set_of_nodes) #if topoloogy is a set
			else:
				N_1,N_2 = node_partition(set_of_nodes) #if topolog is a graph
			
			bipartitioning_mapping(G_1,N_1)
			bipartitioning_mapping(G_2,N_2)

		if len(task_graph) < 256:
			bipartitioning_mapping(task_graph,processor_graph) #if topology is a graph
		else:
			bipartitioning_mapping(task_graph,processors_set) # if topology is a set
		return mapping		 						


class Topology_aware(Mapping):
	"""Class for the topology aware mapping"""
	def map(self):
		mapping={}
		if len(self.process_graph) > len(self.topology):
			raise ValueError("Not applicable for #proc > #nodes")
		task_set = set(self.process_graph.nodes())
		processor_set = {i for i in range(len(self.topology))}
		processor_set_copy = copy.copy(processor_set)
		task_set_closed = set()
		N = len(task_set)
		f_est_min = {}


		def _to_xyz(index,dim):
			dx ,dy , dz = dim
			x = (index % (dx*dy)) % dx
			y = (index % (dx*dy)) // dx
			z = index // (dx*dy)
			return (x,y,z)
		
		def estimation_func(task,processor):#here we calculate a distance as follows:
			#distance among neighborhood is equal to 1, otherwise, we find a manhattan distance between two processors
			first_approx=0
			#processor_xyz = self.topology.to_xyz(processor)
			processor_xyz = _to_xyz(processor,self.topology.dim)

			for tj in task_set_closed:
				if self.process_graph.has_edge(task,tj):
					if self.topology.are_neighbors(processor_xyz,mapping[tj]):
						distance = 1
					else:
						distance = manhattan_distance(processor_xyz,mapping[tj])
					first_approx += self.process_graph.get_edge_data(task,tj)['weight']*distance
			return first_approx

		def estimation_func_second_approx(task,processor):
			second_approx = 0
			sum_distance = 0
			processor_xyz = self.topology.to_xyz(processor)
			for pj in processor_set_copy:
				pj_xyz = self.topology.to_xyz(pj)
				if self.topology.are_neighbors(processor_xyz,pj_xyz):
					distance = 1
				else:
					distance =  manhattan_distance(processor_xyz,pj_xyz)
				sum_distance += distance
			for tj in task_set:
				if self.process_graph.has_edge(task,tj):
					second_approx += self.process_graph.get_edge_data(task,tj)['weight']
			first_approx=0
			for tj in task_set_closed:
				if self.process_graph.has_edge(task,tj):
					if self.topology.are_neighbors(processor_xyz,mapping[tj]):
						distance = 1
					else:
						distance = manhattan_distance(processor_xyz,mapping[tj])
					first_approx += self.process_graph.get_edge_data(task,tj)['weight']*distance				
			return first_approx+second_approx*sum_distance/N

		def estimation_func_third_approx(task,processor):
			second_approx = 0
			sum_distance = 0
			processor_xyz = self.topology.to_xyz(processor)
			for pj in processor_set_copy:
				pj_xyz = self.topology.to_xyz(pj)
				if self.topology.are_neighbors(processor_xyz,pj_xyz):
					distance = 1
				else:
					distance = manhattan_distance(processor_xyz,pj_xyz)
				sum_distance += distance
			for tj in task_set:
				if self.process_graph.has_edge(task,tj):
					second_approx += self.process_graph.get_edge_data(task,tj)['weight']
			first_approx = 0
			sum_third_distance = 0
			for pj in processor_set:
				pj_xyz = self.topology.to_xyz(pj)
				if self.topology.are_neighbors(processor_xyz,pj_xyz):
					distance = 1
				else:
					distance = manhattan_distance(processor_xyz,pj_xyz)
				sum_third_distance += distance
			sum_third_distance = sum_third_distance/len(processor_set)
			for tj in task_set_closed:
				if self.process_graph.has_edge(task,tj):
					first_approx += self.process_graph.get_edge_data(task,tj)['weight']
			return first_approx*sum_third_distance+second_approx*sum_distance/N


		def criticality(task,k):
			min_estimation = float("inf")
			sum_criticality = 0
			for p in processor_set:
				f_est = estimation_func(task,p)#the fastest one
				#f_est = estimation_func_second_approx(task,p)
				#f_est = estimation_func_third_approx(task,p)#the lowest approximation

				if f_est < min_estimation:
					min_estimation = f_est
					f_est_min[task] = p
				sum_criticality += f_est
			return sum_criticality/(N-k)-min_estimation

		for k in range(N):
			max_criticality = -float("inf")
			for task in task_set:
				critical_t = criticality(task,k)
				if critical_t > max_criticality:
					tk = task
					max_criticality = critical_t
			pk = f_est_min[tk]
			mapping[tk]=self.topology.to_xyz(pk)
			task_set_closed.add(tk)
			task_set.remove(tk)
			processor_set.remove(pk)
		return mapping	

class Random(Mapping):
	"""Class for the random sampling mapping"""

	def map(self):
		best = 0
		mapping  = {}
		num_nodes, num_procs= len(self.topology), len(self.process_graph)
		for i in range(10000):
			current = {process: self.topology.to_xyz(node) for process, node in enumerate(sample(range(num_nodes), num_procs))}
			if self.cardinality(current) > best:
				mapping, best = current, self.cardinality(current)
		return mapping


class MinimumManhattanDistance(Mapping):
	"""Class for the minimum manhattan distance mapping"""

	def map(self):
		mapping = {}

		def helper(edge):
			"""
			Helper function for sorting of process pairs.
			:param edge: Process pair
			:return: Sorting value
			"""
			_, _, data = edge
			if 'weight' in data:
				return data['weight']
			else:
				# HAEC Communications Models (Internal Documentation)
				# 2.6 Influence of Message Size on Transfer Time
				if data['size']/data['count'] > 7295:
					return (0, data['size'])
				else:
					return (1, data['size'])

		edge_list = sorted(self.process_graph.edges(data=True),
						   key=helper, reverse=True)
		start_node = tuple((i // 2 for i in self.topology.dim))
		procs_on_node = {}

		def bfs(root):
			"""
			Breadth-first search for a free node
			:param root: Start node
			:return: Next free node
			"""
			max_procs = 1
			def is_full(pos):
				return procs_on_node.get(pos, 0) >= max_procs
			while True:
				if not is_full(root):
					procs_on_node[root] = procs_on_node.get(root, 0) + 1
					return root
				_open = deque([root])
				closed = set([root])
				while len(_open) > 0:
					x, y, z = _open.pop()
					for neighbour in self.topology.neighbours(x, y, z):
						if not is_full(neighbour):
							procs_on_node[neighbour] = procs_on_node.get(neighbour, 0) + 1
							return neighbour
						if neighbour not in closed:
							closed.add(neighbour)
							_open.append(neighbour)
				max_procs += 1  # increase max_proc if no place is found

		while len(edge_list) > 0:
			sender, receiver, weight = edge_list.pop()
			if sender in mapping:
				if receiver in mapping:
					continue
				mapping[receiver] = bfs(mapping[sender])
			elif receiver in mapping:
				mapping[sender] = bfs(mapping[receiver])
			else:
				mapping[sender] = bfs(start_node)
				mapping[receiver]= bfs(mapping[sender])

		return mapping


class SpaceFillingCurve(Mapping):
	"""Base class for space-filling curves"""

	def map(self):
		mapping = {}
		spc_path = self.path(*self.topology.size)
		process_list = self.process_list(self.process_graph, self.topology, self.block)
		for sublist in process_list:
			try:
				cord = next(spc_path)
			except StopIteration: # if num proc > num nodes --> get new generator
				spc_path = self.path(*self.topology.size)
				cord = next(spc_path)
			for proc in sublist:
				mapping[proc] = cord
		return mapping

	def split_evenly(self, _list, num_bins):
		"""
		Split a list into evenly sized number of bins
		:param _list: List
		:param num_bins: Number of bins
		:return: Splitted list
		"""
		index, count = 0, len(_list)
		bin_size, modulo = count // num_bins, count % num_bins
		for i in range(num_bins):
			step = bin_size + 1 if i < modulo else bin_size
			yield _list[index:index+step]
			index += step

	def process_list(self, process_graph, topology, block):
		"""
		Returns the processes in the process graph either grouped or alone.
		:param process_graph: Process graph
		:param topology: Topology
		:param block: Block assignment
		:return:
		"""
		process_list = range(len(process_graph))
		if block:
			return self.split_evenly(process_list, len(topology))
		else:
			return [[proc] for proc in process_list]

	def check_dimensions(self, dx, dy, dz):
		"""
		Checks if dimensions are appropriate for SFC
		:param dx: Number of nodes in x direction
		:param dy: Number of nodes in y direction
		:param dz: Number of nodes in z direction
		"""
		def is_power_of_2(num):
			return (num & (num-1)) == 0 and num != 0
		if dx != dy or dx != dz:
			raise ValueError("Mapping works only for equal dx, dy, dz.")
		if not is_power_of_2(dx):
			raise ValueError("Mapping works only if dx, dy, dz are a powers of 2.")


class Sweep(SpaceFillingCurve):
	"""Class for the Sweep space-filling curve"""

	def path(self, dx, dy, dz):
		"""Returns a generator for the Sweep SFC
		:param dx: Number of nodes in x direction
		:param dy: Number of nodes in y direction
		:param dz: Number of nodes in z direction
		:return: Sweep SFC generator
		"""
		for z in range(dz):
			for y in range(dy):
				for x in range(dx):
					yield (x, y, z)


class Scan(SpaceFillingCurve):
	"""Class for the Scan space-filling curve"""

	def path(self, dx, dy, dz):
		"""Returns a generator for the Scan SFC
		:param dx: Number of nodes in x direction
		:param dy: Number of nodes in y direction
		:param dz: Number of nodes in z direction
		:return: Scan SFC generator
		"""
		x_flip, y_flip = False, False
		for z in range(dz):
			for y in range(dy-1, -1, -1) if y_flip else range(dy):
				for x in range(dx-1, -1, -1) if x_flip else range(dx):
					yield (x, y, z)
				x_flip = not x_flip
			y_flip = not y_flip

#What I added

class Diagonal(SpaceFillingCurve):
	"""Class for the Diagonal space-filling curve (zig zag)"""
	
	def path(self, dx, dy, dz, direction=True):
		"""Returns a generator for the Diagonal SFC
		:param dx: Number of nodes in x direction
		:param dy: Number of nodes in y direction
		:param dz: Number of nodes in z direction
		:param direction: A direction of a zigzag line-
						  True: first node after start (0,0,0) position is on Oy axe
						  False: first node after start (0,0,0) position is on Ox axe
		:return: Diagonal SFC generator
		""" 
		for z_coord in range(dz):
			x_coord,y_coord = 0,0
			dimension = dx*dy
			while True:
				even = ((x_coord+y_coord)%2 == 0)
				yield (x_coord,y_coord,z_coord)
		
				if (even == direction):			 
					x_coord-=1 
					y_coord+=1
					if (y_coord==dy):
						y_coord-=1	
						x_coord+=2 
					if (x_coord<0):
						x_coord=0
				else:
					x_coord+=1
					y_coord-=1
					if (x_coord==dx):
						x_coord-=1
						y_coord+=2
					if (y_coord<0):
						y_coord=0
				dimension-=1
				if (dimension<=0):
					break


class Peano(SpaceFillingCurve):
	"""Class for the Peano space-filling curve"""

	def path(self, dx, dy, dz):
		"""Returns a generator for the Peano SFC
		:param dx: Number of nodes in x direction
		:param dy: Number of nodes in y direction
		:param dz: Number of nodes in z direction
		:return: Peano SFC generator
		"""
		self.check_dimensions(dx, dy, dz)
		def peano(o, u, v, w):
			if u.norm < 1 and v.norm < 1 and w.norm < 1:
				yield o.value
			else:
				for p in [peano(o, u//2, v//2, w//2),
						  peano(o+u-u//2, u//2, v//2, w//2),
						  peano(o+v-v//2, u//2, v//2, w//2),
						  peano(o+u-u//2+v-v//2, u//2, v//2, w//2),
						  peano(o+w-w//2, u//2, v//2, w//2),
						  peano(o+u-u//2+w-w//2, u//2, v//2, w//2),
						  peano(o+v-v//2+w-w//2, u//2, v//2, w//2),
						  peano(o+u-u//2+v-v//2+w-w//2, u//2, v//2, w//2)]:
					for i in p:
						yield i

		o = Vector(0, 0, 0)
		u = Vector(dx-1, 0, 0)
		v = Vector(0, dy-1, 0)
		w = Vector(0, 0, dz-1)
		return peano(o, u, v, w)


class Gray(SpaceFillingCurve):
	"""Class for the Gray space-filling curve"""

	def path(self, dx, dy, dz):
		"""Returns a generator for the Gray SFC
		:param dx: Number of nodes in x direction
		:param dy: Number of nodes in y direction
		:param dz: Number of nodes in z direction
		:return: Gray SFC generator
		"""
		self.check_dimensions(dx, dy, dz)

		def gray(o, u, v, w):
			if u.norm < 1 and v.norm < 1 and w.norm < 1:
				yield o.value
			else:
				for g in [gray(o, u//2, v//2, w//2),
						  gray(o+u+v//2, -u//2, -v//2, w//2),
						  gray(o+u+w, -u//2, v//2, -w//2),
						  gray(o+v//2+w, u//2, -v//2, -w//2),
						  gray(o+v+w, u//2, -v//2, -w//2),
						  gray(o+u+v-v//2+w, -u//2, v//2, -w//2),
						  gray(o+u+v, -u//2, -v//2, w//2),
						  gray(o+v-v//2, u//2, v//2, w//2)]:
					for i in g:
						yield i

		o = Vector(0, 0, 0)
		u = Vector(dx-1, 0, 0)
		v = Vector(0, dy-1, 0)
		w = Vector(0, 0, dz-1)
		return gray(o, u, v, w)


class Hilbert(SpaceFillingCurve):
	"""Class for the Hilbert space-filling curve"""

	def path(self, dx, dy, dz):
		"""
		Returns a generator for the Hilbert SFC
		:param dx: Number of nodes in x direction
		:param dy: Number of nodes in y direction
		:param dz: Number of nodes in z direction
		:return: Hilbert SFC generator
		"""
		"""after ref08.pdf, page 110, Fig. 8.2, right
		"""
		self.check_dimensions(dx, dy, dz)
		def hilbert(o, u, v, w):
			if u.norm < 1 and v.norm < 1 and w.norm < 1:
				yield o.value
			else:
				for h in [hilbert(o, w//2, u//2, v//2),
						  hilbert(o+v-v//2, v//2, w//2, u//2),
						  hilbert(o+u-u//2+v-v//2, v//2,w//2,u//2),
						  hilbert(o+u+v//2, -u//2, -v//2, w//2),
						  hilbert(o+u+v//2+w-w//2, -u//2,-v//2,w//2),
						  hilbert(o+u+v-v//2+w, v//2, -w//2, -u//2),
						  hilbert(o+u//2+v-v//2+w, v//2,-w//2,-u//2),
						  hilbert(o+v//2+w, -w//2, u//2, -v//2)]:
					for i in h:
						yield i

		o = Vector(0, 0, 0)
		u = Vector(dx-1, 0, 0)
		v = Vector(0, dy-1, 0)
		w = Vector(0, 0, dz-1)
		return hilbert(o, u, v, w)


import time


class Bokhari(Mapping):
	"""Class which represents Bokhari's mapping"""

	def get_initial_mapping(self, dim):
		"""
		Returns an initial mapping (Sweep SFC) for the algorithm
		:param dim: Dimensions of topology
		:return: Mapping
		"""
		dx, dy, dz = dim
		return {i: (x, y, z) for i, (x, y, z)
				in enumerate(product(range(dx), range(dy), range(dz)))}


	def map(self):
		if len(self.process_graph) > len(self.topology):
			raise ValueError("Not applicable for #proc > #nodes")

		mapping = self.get_initial_mapping(self.topology.dim)
		best, done = mapping.copy(), False
		self.logger.info('Initial cardinality: %d' % self.cardinality(mapping))
		while not done:
			augmented, count = False, 0
			while True:
				augmented = self.augment(mapping)
				self.logger.info('Cardinality - Round %d: %d' % (count, self.cardinality(mapping)))
				if not augmented:
					break
				count += 1
			if self.cardinality(mapping) < self.cardinality(best):
				done = True
			else:
				best = mapping.copy()
				self.jump(mapping)
		self.logger.info('Cardinality best: %d' % self.cardinality(best))
		return best

	def gain(self, u, v, old, new):
		"""
		Returns the gain of the swap of two processes
		:param u: Node u
		:param v: Node v
		:param old: Old mapping
		:param new: New mapping
		:return: Gain in cardinality
		"""
		u_neighbors = self.process_graph.neighbors(u)
		v_neighbors = self.process_graph.neighbors(v)
		old_card = sum([1 for i in u_neighbors if self.topology.are_neighbors(old[u], old[i])]) + sum([1 for j in v_neighbors if self.topology.are_neighbors(old[v], old[j])])
		new_card = sum([1 for i in u_neighbors if self.topology.are_neighbors(new[u], new[i])]) + sum([1 for j in v_neighbors if self.topology.are_neighbors(new[v], new[j])])

		return new_card - old_card

	def augment(self, mapping):
		"""
		Augments mapping based on pairwise swaps
		:param mapping: Initial mapping
		:return: Augmented mapping
		"""
		improved = False
		for i, u in enumerate(self.process_graph):
			if i % 100 == 0:
				self.logger.info('Augment u: %d' % i)
			ex_pair, max_gain = None, 0
			for j, v in enumerate(self.process_graph):
				tmp_mapping = mapping.copy()
				if u == v:
					continue
				tmp_mapping[u], tmp_mapping[v] = tmp_mapping[v], tmp_mapping[u]
				gain = self.gain(u, v, mapping, tmp_mapping)
				if gain >= max_gain:
					ex_pair, max_gain = (u, v), gain
			if max_gain > 0:
				improved = True
			if ex_pair is not None:
				u, v = ex_pair
				mapping[u], mapping[v] = mapping[v], mapping[u]
		return improved

	def jump(self, mapping):
		for i in range(max(self.topology.dim)):
			u, v = sample(range(len(self.process_graph)), 2)
			mapping[u], mapping[v] = mapping[v], mapping[u]


class StateSpaceSearch(Mapping):
	"""
	Defines a base class and methods for a state-space search
	"""

	def map(self):
		_open, closed = [self.SearchNode(self, dict())], set()
		while len(_open) != 0:
			node = heappop(_open)
			if node not in closed:
				closed.add(node)
				if self.is_goal(node):
					return node.mapping
				for s in self.succ(node):
					heappush(_open, s)

		raise ValueError("StateSpaceSearch: Not found a solution")

	class SearchNode():
		def __init__(self, outer, mapping):
			self.outer = outer
			self.mapping = mapping

		def __eq__(self, other):
			return set(self.mapping.items()) == set(other.mapping.items())

		def __lt__(self, other):
			g_self, g_other = self.outer.g(self.mapping), self.outer.g(other.mapping)
			h_self, h_other = self.outer.h(self.mapping), self.outer.h(other.mapping)
			if g_self + h_self == g_other + h_other:
				return h_self < h_other
			else:
				return g_self + h_self < g_other + h_other


		def __hash__(self):
			return hash(frozenset(self.mapping.items()))

	def md(self, u, v):
		return sum([abs(i - j) for i, j in zip(u, v)])

	def is_goal(self, node):
		return set(self.process_graph) == set(node.mapping)


class Genetic(Mapping):

	population_size = 100
	generations = 10

	def fitness(self, string):
		"""
		Returns the fitness of a string.
		:param string: String
		:return: Fitness
		"""
		_sum = 0
		for u, v in self.process_graph.edges():
			cords_u = self.topology.to_xyz(string[u])
			cords_v = self.topology.to_xyz(string[v])
			if self.topology.are_neighbors(cords_u, cords_v):
				_sum += 1
		return _sum

	def selection(self, population):
		"""
		Selects a candidate in the population (roulette wheel selection)
		After: http://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
		:param population: Candidate list
		:return: Candidate
		"""
		_sum = sum(population.values())
		pick = uniform(0, _sum)
		current = 0
		for string, fitness in population.items():
			current += fitness
			if current > pick:
				return string

	def crossover(self, a, b):
		"""
		Executes a an ordered crossover
		After: Wonjae Lee and Hak-Young Kim, "Genetic algorithm implementation in Python," Fourth Annual ACIS International Conference on Computer and Information Science (ICIS'05), 2005, pp. 8-11.
		:param a: Parent
		:param b: Parent
		:return: Offspring
		"""
		a, b = list(a), list(b)
		p, q = sample(range(1, len(self.topology)-1), 2)
		p, q = min(p, q), max(p, q)

		cross_a, cross_b = a[p:q], b[p:q]
		remain_a, remain_b = a[q:] + a[:q], b[q:] + b[:q]

		filter_a = [i for i in remain_a if i not in cross_b]
		filter_b = [i for i in remain_b if i not in cross_a]

		c = filter_a[-p:] + cross_b + filter_a[:len(a)-q]
		d = filter_b[-p:] + cross_a + filter_b[:len(b)-q]

		return tuple(c), tuple(d)

	def mutation(self, a):
		"""
		Implements a mutation in a string
		:param a: String
		:return:
		"""
		a = list(a)
		p = randint(0, len(self.topology) - 1)
		q = randint(0, len(self.topology) - 1)
		a[p], a[q] = a[q], a[p]
		return tuple(a)

	def create_generation(self, population):
		"""
		Creates a new generation based on a population
		:param population: Population
		:return: Generation
		"""
		generation, size = {}, len(population)
		while len(generation) < size and (time.time()-self.t < 10800):
			a, b = self.selection(population), self.selection(population)
			c, d = self.crossover(a, b)
			c = self.mutation(c)
			d = self.mutation(d)
			fit_a, fit_b = self.fitness(a), self.fitness(b)
			fit_c, fit_d = self.fitness(c), self.fitness(d)
			if max(fit_a, fit_b) < fit_c:
				generation[c] = fit_c
			if max(fit_a, fit_b) < fit_d:
				generation[d] = fit_d

		return generation

	def initial_population(self):
		num_nodes = len(self.topology)
		population = {}
		for _ in range(self.population_size):
			s = sample(range(num_nodes), num_nodes)
			population[tuple(s)] = self.fitness(s)
		return population

	def to_mapping(self, string):
		return {process: self.topology.to_xyz(node) for process, node in enumerate(string)}

	def map(self):
		population = self.initial_population()
		self.t = time.time()
		fittest = max(population, key=population.get)
		print(self.fitness(fittest))
		for i in range(self.generations):
			print(i)
			generation = self.create_generation(population)
			if generation:
				population = generation
		fittest = max(population, key=population.get)
		print(self.fitness(fittest))
		return self.to_mapping(fittest)
