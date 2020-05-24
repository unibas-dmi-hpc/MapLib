from .routing import route, get_edges, group_edges, manhattan_distance
from .topology import Mesh, Torus, Haec

def ieplc(mapping):
    """
    Returns the statistics for the number of inter-process logical communications (IePLC).
    :param mapping: process-to-node mapping
    :return: IePLC statistics
    """
    weights = [d['weight'] for u, v, d in mapping.process_graph.edges(data=True)]
    return sum(weights), sum(weights)//len(weights) + 1, min(weights), max(weights)

def ianlc(mapping):
    """
    Returns the statistics for the number of intra-node logical communications (IaNLC).
    :param mapping: process-to-node mapping
    :return: IaNLC statistics
    """
    weights = [d['weight'] for u, v, d in mapping.process_graph.edges(data=True)
               if mapping.mapping[u] == mapping.mapping[v]]
    try:
        return sum(weights), sum(weights)//len(weights) + 1, min(weights), max(weights)
    except ZeroDivisionError:
        return (0, 0, 0, 0)

def ienlc(mapping):
    """
    Returns the statistics for the number of inter-node logical communications (IeNLC).
    :param mapping: process-to-node mapping
    :return: IeNLC statistics
    """
    edges = [(u, v, w) for u, v, w in get_edges(mapping) if u != v]
    weights = [w for u, v, w in group_edges(edges)]
    return sum(weights), sum(weights)//len(weights) + 1, min(weights), max(weights)

def ienpc(mapping):
    """
    Returns the statistics for the number of inter-node physical communications (IaNPC).
    :param mapping: process-to-node mapping
    :return: IeNPC statistics
    """
    edges = group_edges(route(mapping))
    IeNPC=0
    for i,j,w in mapping.process_graph.edges(data=True):
    	first_node = mapping.mapping[i]
    	second_node = mapping.mapping[j]
    	IeNPC += w['weight']*mapping.topology.hops(first_node,second_node)
    weights = [w for u, v, w in edges ]
    if not isinstance(mapping._topology,Mesh):
    	IeNPC = sum(weights)
    return IeNPC, sum(weights)//len(weights) + 1, min(weights), max(weights)

def wtad_default(mapping):
	#Task#1 is placed onto the processor #1, #2 -> #2 and so on - it is a mapping by default#
	_sum = 0
	cardinality = len(mapping.process_graph)
	for a in range(len(mapping.topology)):
		a_xyz = mapping.topology.to_xyz(a)
		for b in range(len(mapping.topology)):
			distance = manhattan_distance(a_xyz,mapping.topology.to_xyz(b))
			if mapping.process_graph.has_edge(a,b):
				W = mapping.process_graph.get_edge_data(a,b)['weight']
				_sum += distance*W
	return _sum//(cardinality**2-1)



def wtad(mapping):
    reverse_map =  {v: k for k, v in mapping.mapping.items()}
    new_map = {mapping.topology.to_idx(key):value for key,value in reverse_map.items()}
    cardinality = len(mapping.process_graph)
    _sum = 0
    #min_sum = float("inf")
    #max_sum = 0
    #col = 0
    for a in range(len(mapping.topology)):
        a_xyz = mapping.topology.to_xyz(a)
        for b in range(len(mapping.topology)):
            distance = manhattan_distance(a_xyz,mapping.topology.to_xyz(b))
            if mapping.process_graph.has_edge(new_map[a],new_map[b]):
                W = mapping.process_graph.get_edge_data(new_map[a],new_map[b])['weight']
                #if distance*W > max_sum:
                #    max_sum = distance*W
                #if distance*W < min_sum:
                #    min_sum = distance*W 
                _sum += distance*W
                #col += 1
    return _sum//(cardinality**2-1)#, _sum // col, min_sum, max_sum 

def wtad_al(mapping):
    reverse_map =  {v: k for k, v in mapping.mapping.items()}
    new_map = {mapping.topology.to_idx(key):value for key,value in reverse_map.items()}
    cardinality = len(mapping.process_graph)
    _sum = 0
    #min_sum = float("inf")
    #max_sum = 0
    #col = 0

    def asym(a,b):
        (ax,ay,az), (bx,by,bz) = a,b
        return abs((ax-bx) - (ay-by) - (az-bz))

    for a in range(len(mapping.topology)):
        a_xyz = mapping.topology.to_xyz(a)
        for b in range(len(mapping.topology)):
            distance = manhattan_distance(a_xyz,mapping.topology.to_xyz(b)) + asym(a_xyz,mapping.topology.to_xyz(b))
            if mapping.process_graph.has_edge(new_map[a],new_map[b]):
                W = mapping.process_graph.get_edge_data(new_map[a],new_map[b])['weight']
                #if distance*W > max_sum:
                #    max_sum = distance*W
                #if distance*W < min_sum:
                #    min_sum = distance*W 
                _sum += distance*W
                #col += 1
    return _sum//(cardinality**2-1)#, _sum//col, min_sum, max_sum 

def NA(mapping):
    return len(mapping.topology)

def AD(mapping):
    cardinality = len(mapping.process_graph)
    distance = 0
    for a in range(len(mapping.topology)):
        a_xyz = mapping.topology.to_xyz(a)
        for b in range(len(mapping.topology)):
            distance += manhattan_distance(a_xyz,mapping.topology.to_xyz(b))
    return distance//(cardinality*(cardinality-1))

def DFC(mapping):
    distance = []
    for a in range(len(mapping.topology)):
        a_xyz = mapping.topology.to_xyz(a)
        _sum = 0
        for b in range(len(mapping.topology)):
            _sum += manhattan_distance(a_xyz,mapping.topology.to_xyz(b))
        distance.append(_sum)
    return min(distance)

def Diam(mapping):
    _max = []
    for a in range(len(mapping.topology)):
        a_xyz = mapping.topology.to_xyz(a)
        for b in range(len(mapping.topology)):
            _max.append(manhattan_distance(a_xyz,mapping.topology.to_xyz(b)))
    return max(_max)

def LA(mapping):
    '''DISCLAIMER: According to formula: a difference between max and min is just how many coordinates in each dimension we have
       In our case, we have a cube, hence, dx=dy=dz, so, max-min is just dx-1 (or = dz-1 or = dy-1, no matter)
       Multiplication: how many allocated node we have. because dx=dy=dz, we can choose just one of them
       And factor 3 is just because we sum up 3 equal thing'''
    '''LA shows how many affected links we have in the topology in case of all-to-all communication'''
    return 3*((mapping.topology.dx-1)*mapping.topology.dx**2) 









