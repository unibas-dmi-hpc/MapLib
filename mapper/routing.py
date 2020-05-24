from itertools import groupby

from .topology import Mesh, Torus, Haec


def left(u, i, d):
    return tuple((k + d + 1) % d if j == i else k for j, k in enumerate(u))


def right(u, i, d):
    return tuple((k + d - 1) % d if j == i else k for j, k in enumerate(u))


def xyz_path(u, v, dim, topology):
    """
    Returns the path from node u to v using xyz routing.
    :param u: Node
    :param v: Node
    :param dim: Topology dimensions
    :param topology: Topology
    :return: Edge list
    """
    if u == v:
        yield u, v
    elif isinstance(topology, (Mesh, Torus)) or (isinstance(topology, Haec) and u[2] == v[2]):
        for i, d in enumerate(dim):
            if isinstance(topology, (Torus, Haec)) and abs(u[i] - v[i]) <= d // 2:
                while u[i] < v[i]:
                    u_old, u = u, left(u, i, d)
                    yield u_old, u
                while u[i] > v[i]:
                    u_old, u = u, right(u, i, d)
                    yield u_old, u
            else:
                while u[i] < v[i]:
                    u_old, u = u, right(u, i, d)
                    yield u_old, u
                while u[i] > v[i]:
                    u_old, u = u, left(u, i, d)
                    yield u_old, u
    elif isinstance(topology, Haec):
            while u[2] < v[2]:
                u_old, u = u, (v[0], v[1], u[2] + 1)
                yield u_old, u
            while u[2] > v[2]:
                u_old, u = u, (v[0], v[1], u[2] - 1)
                yield u_old, u
    else:
        raise NotImplementedError



def manhattan_distance(i, j):
    """
    Returns the Manhattan distance between two nodes i and j
    :param i: Node
    :param j: Node
    :return: Manhattan distance
    """
    (ix, iy, iz), (jx, jy, jz) = i, j
    return abs(ix - jx) + abs(iy - jy) + abs(iz - jz)


def get_edges(mapping):
    """
    Returns the sorted edge list for topology.
    :param mapping: Process-to-node mapping
    :return: Topology edge list
    """
    proc_map = mapping.mapping
    edges = mapping.process_graph.edges(data=True)
    return sorted([(min(proc_map[u], proc_map[v]),
                    max(proc_map[u], proc_map[v]),
                    data['weight']) for u, v, data in edges])


def group_edges(edges):
    """
    Performs an aggregation (sum) of edges with equal vertices.
    :param edges: Edge list
    :return: Aggregated edge list
    """
    return [(u, v, sum(w for _, _, w in group)) for (u, v), group
            in groupby(sorted(edges), key=lambda x: (x[0],x[1]))]


def route(mapping):
    """
    Routes the logical edges in the topology to physical edges.
    :param mapping: Process-to-node-mapping
    :return: Routed edge list
    """
    edges = get_edges(mapping)
    dim = mapping.topology.dim
    for u, v, w in edges:
        if manhattan_distance(u, v) > 1:
            for i, j in xyz_path(u, v, dim, mapping.topology):
                yield (i, j, w)
        else:
            yield (u, v, w)
