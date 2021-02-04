from collections import deque
from heapq import heappop, heappush
from itertools import product
from random import sample, randint, random, uniform
import copy
import nxmetis
import networkx as nx
import sys
import math
import bisect

from .vector import Vector
from .routing import manhattan_distance
from .mapping import Mapping

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

def _to_xyz(index,dim):
	dx, dy, dz = dim
	x = (index % (dx*dy)) % dx
	y = (index % (dx*dy)) // dx
	z = index // (dx*dy)
	return (x,y,z)

def Center_node_selection(processor_graph,dim):
	NS = []
	R = math.floor(len(processor_graph)**(1/3)) + 1
	for n in processor_graph.nodes():
		dist = 0
		available_nodes = [i[0] for i in sorted(nx.single_source_dijkstra_path_length(processor_graph,n,R).items(),key=lambda item:item[1])]
		for i in available_nodes:
			try:
				#dist += 1/(nx.dijkstra_path_length(processor_graph,n,i)**3)
				dist += 1/(manhattan_distance(_to_xyz(n,dim),_to_xyz(i,dim))**3)
			except ZeroDivisionError:
				dist += 0
		NS.append(dist)
	return NS.index(max(NS))

def closest_free_nodes(dijkstra_path,start,how_many,process_set):
	free_node = []
	path = [i[0] for i in sorted(dijkstra_path[start].items(),key=lambda item:item[1])]
	for elem in path:
		if elem != start and elem in process_set:
			free_node.append(elem)
		if len(free_node) == how_many:
			break
	return free_node


def next_free_node (path,process_set):
	for elem in path:
		if elem in process_set:
			return elem
	#return free_node

def insertion_sort(array):
	for i in range(1, len(array)):
		while i > 0 and array[i].f < array[i - 1].f:
			array[i], array[i - 1] = array[i - 1], array[i]
			i -= 1
	return array


class Node():
	def __init__(self, name, node, level, parent, g, h, f, successors, local_task_set, local_process_set):
		self.name = name
		self.parent = parent
		self.node_where_placed = node
		self.level = level
		self.g = g
		self.h = h
		self.f = f
		self.successors = successors
		self.local_task_set = local_task_set
		self.local_process_set = local_process_set

class One_to_One(Mapping):
	def map(self):
		mapping = {i:self.topology.to_xyz(i) for i in range(len(self.process_graph))}
		return mapping




class A_star_Ali(Mapping):
	"""Class for A* mapping proposed by Ali"""
	def map(self):
		task_graph = self.process_graph
		processor_graph = nx.Graph()
		for elem in range(len(self.topology)):
			x,y,z = self.topology.to_xyz(elem)
			for neighbour in self.topology.neighbours(x,y,z):
				processor_graph.add_edge(elem,self.topology.to_idx(neighbour))
		S_opt = int(task_graph.size(weight='weight'))*2
		#print(processor_graph.edges())
		#S_opt = 62

		mapping = {}
		return_mapping = {}
		Hopes = [[0]*len(processor_graph) for i in range(len(processor_graph))]
		for i in processor_graph.nodes():
			for j in processor_graph.nodes():
				Hopes[i][j] = nx.dijkstra_path_length(processor_graph,i,j)
		print(Hopes)

		task_set = {i for i in task_graph.nodes()}
		processor_set = {i for i in processor_graph.nodes()}

		path = nx.all_pairs_dijkstra_path_length(processor_graph)
		for key in path.keys(): #delete all elements with 0-distance
			for elem in list(path[key].keys()):
				if elem == key:
					del path[key][elem]

		sort_path = {}
		for node in processor_graph.nodes():
			sort_path[node] = [i[0] for i in sorted(path[node].items(),key=lambda item:item[1])]

		start_node = sorted(task_graph.degree(None,'weight').items(), key=lambda item:item[1], reverse=True).pop(0)[0]
		next_free = Center_node_selection(processor_graph,self.topology.dim)
		#next_free = len(processor_graph.nodes()) // 2
		#next_free = 0
		#print(next_free)

		mapping[start_node] = next_free

		task_set.remove(start_node)
		processor_set.remove(next_free)

		open_set = []
		closed_nodes = []

		Rem_Unmap_Conn = int(task_graph.size(weight='weight'))
		successors = set()
		for elem in task_graph.neighbors(start_node):
			successors.add((elem,start_node,next_free))

		#bisect.insort(open_set,(0+Rem_Unmap_Conn,start_node,next_free,1,None,0,Rem_Unmap_Conn,successors,task_set,processor_set))
		#name, node, level, parent, g, h, f, successors, local_task_set, local_process_set
		open_set.append(Node(start_node,next_free,1,None,0,Rem_Unmap_Conn,0+Rem_Unmap_Conn,successors,task_set,processor_set))

		last = 0
		col = 0
		while len(open_set) != 0:

			#next_node = open_set.pop(0)
			#next_node = min(open_set,key=lambda elem:elem.f)
			open_set = insertion_sort(open_set)
			next_node = open_set.pop(0)
			#open_set.remove(next_node)
			#print(next_node.name,next_node.level,next_node.node_where_placed)
			#print(next_node[1],next_node[3],next_node[2])
			#closed_nodes.append(next_node)
			if next_node.level == len(task_graph): #it means that we have assigned all tasks to all processors and come to finish point
				#if next_node.f <= open_set[0].f:
				last = next_node
				break
			#if next_node[3] == len(task_graph):
				#if next_node[0] <= open_set[0][0]:
				#break


			for kid in next_node.successors:
			#for kid in next_node[7]:
				communication = task_graph.get_edge_data(kid[0],kid[1])['weight']
				successors_set = set()
				#free_node = closest_free_nodes(path, kid[2],1,next_node.local_process_set)[0]
				free_node = next_free_node(sort_path[kid[2]],next_node.local_process_set)
				#free_node = next_free_node(sort_path[kid[2]],next_node[9])
				g_n = next_node.g + Hopes[free_node][kid[2]]*communication
				#g_n = next_node[5] + Hopes[free_node][kid[2]]*communication
				h_n = next_node.h - communication
				#h_n = next_node[6] - communication
				if g_n + h_n <= S_opt:
					for elem in next_node.successors-{kid}|{(i,kid[0],free_node) for i in next_node.local_task_set&set(task_graph.neighbors(kid[0]))}:
					#for elem in next_node[7]-{kid}|{(i,kid[0],free_node) for i in next_node[8]&set(task_graph.neighbors(kid[0]))}:
						if elem[0] != kid[0]:
							successors_set.add(elem)
					#bisect.insort(open_set,(g_n+h_n,kid[0],free_node,next_node[3]+1,next_node,g_n,h_n,successors,next_node[8]-{kid[0]},next_node[9]-{free_node}))
					open_set.append(Node(kid[0],free_node,next_node.level+1,next_node,g_n,h_n,g_n+h_n,successors_set,next_node.local_task_set-{kid[0]},next_node.local_process_set-{free_node}))
					#open_set = sorted(open_set,key=lambda elem:elem.f)
					#open_set.sort(key=lambda elem:elem.f)

			#print(col)
			col+=1

		#st = closed_nodes[len(closed_nodes)-1]
		st = last
		while True:
			mapping[st.name] = st.node_where_placed
			if st.parent == None:
				break
			st = st.parent
		'''while True:
			mapping[st[1]] = st[2]
			if st[4] == None:
				break
			st = st[4]'''

		return_mapping = {task:self.topology.to_xyz(xyz) for task,xyz in mapping.items()}
		return return_mapping


		


class A_star(Mapping):
	"""Class for A* mapping algorithm"""
	def map(self):
		task_graph = self.process_graph
		processor_graph = nx.Graph()
		for elem in range(len(self.topology)):
			x,y,z = self.topology.to_xyz(elem)
			for neighbour in self.topology.neighbours(x,y,z):
				processor_graph.add_edge(elem,self.topology.to_idx(neighbour))
		S_opt = int(task_graph.size(weight='weight'))*1


		mapping = {}
		return_mapping = {}
		Hopes = [[0]*len(processor_graph) for i in range(len(processor_graph))]
		for i in processor_graph.nodes():
			for j in processor_graph.nodes():
				Hopes[i][j] = nx.dijkstra_path_length(processor_graph,i,j)

		task_set = {i for i in task_graph.nodes()}
		processor_set = {i for i in processor_graph.nodes()}

		path = nx.all_pairs_dijkstra_path_length(processor_graph)

		start_node = sorted(task_graph.degree(None,'weight').items(),key=lambda item:item[1],reverse=True).pop(0)[0]
		#next_free = Center_node_selection(processor_graph)
		next_free = len(processor_graph.nodes()) // 2
		mapping[start_node] = next_free

		task_set.remove(start_node)
		processor_set.remove(next_free)

		"""A* algorithm"""
		open_set = {}
		open_set[start_node] = {}
		for i in task_graph.edges(start_node):
			open_set[start_node][i[1]] = task_graph.get_edge_data(*i)['weight']

		free_node = {}
		A_set = {}

		while len(task_set) != 0:
			for elem in open_set:
				free_node[elem] = closest_free_nodes(path,mapping[elem],1,processor_set)[0]
				for candidate in open_set[elem]:
					if candidate in task_set:
						g_n = Hopes[mapping[elem]][free_node[elem]] / open_set[elem][candidate]
						_sum = 0
						num_neighbors = 0
						for i in task_graph.neighbors(candidate):
							if i in task_set:
								_sum += task_graph.get_edge_data(i,candidate)['weight']
								num_neighbors += 1
						_sub_free_node = closest_free_nodes(path,free_node[elem],num_neighbors,processor_set-{free_node[elem]})
						_num_hopes = 0
						for i in _sub_free_node:
							_num_hopes += Hopes[free_node[elem]][i]
						if _sum != 0:
							h_n = _num_hopes / _sum
						else:
							h_n = 0
						A_set[(candidate,free_node[elem],elem)] = g_n + h_n
						if g_n + h_n <= S_opt:
							A_set[(candidate,free_node[elem],elem)] = g_n + h_n
			next_pair = sorted(A_set.items(),key=lambda item:item[1]).pop(0)[0]
			mapping[next_pair[0]] = next_pair[1]

			task_set.remove(next_pair[0])
			processor_set.remove(next_pair[1])
			A_set.clear()
			free_node.clear()
			for i in task_graph.neighbors(next_pair[0]):
				if i in task_set:
					open_set.setdefault(next_pair[0],{})[i] = task_graph.get_edge_data(i,next_pair[0])['weight']

			del open_set[next_pair[2]][next_pair[0]]

		return_mapping = {task:self.topology.to_xyz(xyz) for task,xyz in mapping.items()}

		return return_mapping


class Fast_High_Greedy(Mapping):
	"""Class for fast and high quality greedy mapping algorithm"""
	def map(self):
		task_graph = self.process_graph
		processor_graph = nx.Graph()
		for elem in range(len(self.topology)):
			x,y,z = self.topology.to_xyz(elem)
			for neighbour in self.topology.neighbours(x,y,z):
				processor_graph.add_edge(elem,self.topology.to_idx(neighbour))
		
		connectivity = {i:0 for i in task_graph.nodes()}
		mapping = {}
		task_set = {i for i in task_graph.nodes()}
		processor_set = {i for i in processor_graph.nodes()}
		task_set_mapped = set()
		processor_set_mapped = set()
		return_mapping = {} # what we return as a result: {task:xyz_of_node}

		def GetBestNode(t_best):
			if set(task_graph.neighbors(t_best)).isdisjoint(task_set_mapped):
				far_node_queue = {i:float("inf") for i in processor_set}
				for processor in processor_set_mapped:
					for elem in nx.single_source_dijkstra_path_length(processor_graph,processor,None,'1').items():
						if elem[0] in processor_set:
							if elem[1] < far_node_queue.get(elem[0]):
								far_node_queue[elem[0]] = elem[1]
				m_best = [i[0] for i in sorted(far_node_queue.items(),key=lambda item:item[1],reverse=True)].pop(0)
			else:
				near_node_queue = {}
				intersection = set(task_graph.neighbors(t_best)).intersection(task_set_mapped)
				for map_task in intersection:
					for edge in nx.bfs_edges(processor_graph,mapping[map_task]):
						if edge[1] in processor_set:
							near_node_queue[map_task] = edge[1]
							break
				min_WH = float("inf")
				for elem in near_node_queue.items():
					dilation = nx.astar_path_length(processor_graph,elem[1],mapping[elem[0]])
					WH = dilation*task_graph.get_edge_data(t_best,elem[0])['weight']
					if WH < min_WH:
						min_WH, m_best = WH, elem[1]
			return m_best

		#Find the task with MSRV
		#T_msrv = [i[0] for i in sorted(task_graph.degree(None,'weight').items(), key = lambda item:item[1], reverse=True)].pop(0)
		T_msrv = [i[0] for i in sorted(task_graph.degree(None,'weight'), key = lambda item:item[1], reverse=True)].pop(0)
		t_0 = T_msrv
		#m_0 = Center_node_selection(processor_graph)
		m_0 = len(processor_set) // 2

		#Map t_0 to center node in the topology
		mapping[t_0] = m_0

		#Delete t_0 and m_0 from consideration`s sets
		task_set.remove(t_0)
		processor_set.remove(m_0)
		del connectivity[t_0]

		#Add t_0 and m_0 to sets of mapped tasks and nodes, respectively
		task_set_mapped.add(t_0)
		processor_set_mapped.add(m_0)

		#Update connectivity for the tasks in neighborhood(t_0)
		for neighbor in task_graph.neighbors(t_0):
			if neighbor != t_0: #LLP: added line to solve situations with communication loops (edge has the same source and destination)
				connectivity[neighbor] += task_graph.get_edge_data(neighbor,t_0)['weight']

		#Choose a number N_bfs. It denotes the number of tasks to map before starting the greedy graph mapping
		N_bfs = 1
		while len(task_set) != 0:
			if len(task_set_mapped) < N_bfs:
				#Find the farthest unmapped task
				far_task_queue = {i:float("inf") for i in task_set}
				for map_task in task_set_mapped:
					for elem in nx.single_source_dijkstra_path_length(task_graph,map_task,None,'1').items():
						if elem[0] in task_set:
							if elem[1] < far_task_queue.get(elem[0]):
								far_task_queue[elem[0]] = elem[1]
				t_best = [i[0] for i in sorted(far_task_queue.items(),key=lambda item:item[1],reverse=True)].pop(0)
			else:
				#the one with maximum connectivity = {task:connectivity}. Hence since we have to choose a task, then we choose 0 elem of tuple
				t_best = sorted(connectivity.items(),key=lambda item:item[1],reverse=True).pop(0)[0]
			#Choose m_best using GetBestNode
			m_best = GetBestNode(t_best)
			#mapping t_best on m_best
			mapping[t_best] = m_best
			#Add t_best and m_best to mapped tasks and nodes respectively
			task_set_mapped.add(t_best)
			processor_set_mapped.add(m_best)
			#Update connectivity for each neighbor of t_best that are still unmapped
			for neighbor in task_graph.neighbors(t_best):
				if neighbor in task_set:
					connectivity[neighbor] += task_graph.get_edge_data(neighbor,t_best)['weight']
			#Delete t_best and m_best from consideartion`s sets
			del connectivity[t_best]
			task_set.remove(t_best)
			processor_set.remove(m_best)

		return_mapping = {task:self.topology.to_xyz(xyz) for task,xyz in mapping.items()}
		return return_mapping

 

class pacMAP(Mapping):
	"""Class for PaCMap mapping"""
	def map(self):
		task_graph = self.process_graph
		processor_graph = nx.Graph()
		for elem in range(len(self.topology)):
			x,y,z = self.topology.to_xyz(elem)
			for neighbour in self.topology.neighbours(x,y,z):
				processor_graph.add_edge(elem,self.topology.to_idx(neighbour))
		mapping = {}
		processor_set = {i for i in processor_graph.nodes()}
		task_set = {i for i in task_graph.nodes()}

		def Center_task_selection():
			TG = []
			for i in task_graph.nodes():
				TG.append(sum(nx.all_pairs_dijkstra_path_length(task_graph,None,'1')[i].values()))
			return TG.index(min(TG))

		start_node = Center_node_selection(processor_graph,self.topology.dim)
		#start_node = len(processor_graph.nodes()) // 2
		#start_task = Center_task_selection()
		start_task = nx.center(task_graph)[0]
		mapping[start_task] = self.topology.to_xyz(start_node)

		processor_set.remove(start_node)
		task_set.remove(start_task)

		task_set_mapped = set()
		task_set_mapped.add(start_task)

		expansion = {}
		neighbours_max = {}
		
		while len(task_set) != 0:
			val_max_task = 0 #min distance
			max_task =  list(task_set)[0] #some random task
			#neighbours_max = {}
			neighbours_max.clear()
			for task in task_set:
				val = 0
				neighbours = {}
				for map_task in task_set_mapped:
					if task_graph.has_edge(task,map_task):
						weight_of_edge = task_graph.get_edge_data(task,map_task)['weight']
						val += weight_of_edge
						neighbours[map_task] = weight_of_edge
				if val > val_max_task:
					val_max_task, max_task, neighbours_max = val, task, neighbours
				#elif val == val_max_task:
				#	if nx.astar_path_length(task_graph,start_task,task) < nx.astar_path_length(task_graph,start_task,max_task): #if task is near to center than max_task
				#		val_max_task, max_task, neighbours_max = val, task, neighbours
			#expansion = {}
			expansion.clear()
			for node in processor_set:
				overhead = 0
				for neighbour_task in neighbours_max:
					#overhead += nx.dijkstra_path_length(processor_graph,node,self.topology.to_idx(mapping[neighbour_task]))*neighbours_max.get(neighbour_task)
					overhead += manhattan_distance(self.topology.to_xyz(node),mapping[neighbour_task])*neighbours_max.get(neighbour_task)
				expansion[node] = overhead
			max_node = [i[0] for i in sorted(expansion.items(),key=lambda item:item[1])].pop(0)
			mapping[max_task] = self.topology.to_xyz(max_node)
			task_set.remove(max_task)
			processor_set.remove(max_node)
			task_set_mapped.add(max_task)
		return mapping
