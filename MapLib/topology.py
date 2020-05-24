import inspect
import sys


class Topology(object):
    """Base class which represents a node/processor topology. """

    @staticmethod
    def get_topology(name, dimensions):
        for _name, _object in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(_object) and _name.lower() == name.lower():
                return _object(dimensions)
        raise ValueError('Unknown topology: "%s"' % name)

    def __init__(self, dimensions):
        self._dx, self._dy, self._dz = dimensions

    def __len__(self):
        """
        Returns the number of nodes in the topology.
        :return: Number of nodes
        """
        return self._dx * self._dy * self._dz

    @property
    def dx(self):
        """
        Returns the number of nodes in x direction.
        :return: Number of nodes in x direction
        """
        return self._dx

    @property
    def dy(self):
        """
        Returns the number of nodes in y direction.
        :return: Number of nodes in y direction
        """
        return self._dy

    @property
    def dz(self):
        """
        Returns the number of nodes in z direction.
        :return: Number of nodes in z direction
        """
        return self._dz

    @property
    def dim(self):
        """
        Returns a tuple with the dimensions of the topology.
        :return: Topology dimensions
        """
        return self._dx, self._dy, self._dz

    @property
    def size(self):
        """Returns a tuble with the dimensions of the topology."""
        return (self._dx, self._dy, self._dz)

    def to_xyz(self, index):
        """
        Converts an node index i to a coordinate (x,y,z).
        :param index: Node index
        :return: Node coordinate
        """
        if index < 0:
            raise ValueError("Negative index")
        if index >= self.dx * self.dy * self.dz:
            raise ValueError("Index exceeds matrix dimensions")
        x = (index % (self.dx * self.dy)) % self.dx
        y = (index % (self.dx * self.dy)) // self.dx
        z = index // (self.dx * self.dy)
        return (x, y, z)

    def to_idx(self, coordinates):
        """
        Converts a coordinate (x,y,z) to a node index i.
        :param coordinates: Node coordinate
        :return: Node index
        """
        x, y, z = coordinates
        if x < 0 or y < 0 or z < 0:
            raise ValueError("Negative coordinates")
        if x >= self.dx or y >= self.dy or z >= self.dz:
            raise ValueError("Coordinates exceed matrix dimensions")
        return ((z * self.dy) + y) * self.dx + x

    def are_neighbors(self, i, j):
        """
        Returns True if node i and j are neighbors in the topology
        :param i: Node
        :param j: Node
        :return: Are neighbors
        """
        (x, y, z) = i
        return j in self.neighbours(x, y, z)

    def hops(self, i, j):
        """
        Returns the minimal number of hops between i and j by a
        breadth-first search
        :param i: Node
        :param j: Node
        :return: Hops
        """
        queue = []
        closed = set()
        node = (i, [])
        queue.append(node)
        while len(queue) > 0:
            current, parents = queue.pop(0)
            closed.add(current)
            if current == j:
                return len(parents)
            else:
                (x, y, z) = current
                for k in self.neighbours(x, y, z):
                    if k not in closed:
                        queue.append((k, parents + [current]))
        raise ValueError('No path found')


class Mesh(Topology):
    """Class wich defines a 3D-Mesh."""

    def neighbours(self, x, y, z):
        if x < self._dx - 1:
            yield (x + 1, y, z)
        if x > 0:
            yield (x - 1, y, z)
        if y < self._dy - 1:
            yield (x, y + 1, z)
        if y > 0:
            yield(x, y - 1, z)
        if z < self._dz - 1:
            yield (x, y, z + 1)
        if z > 0:
            yield (x, y, z - 1)


class Torus(Topology):
    """Class which defines a 3D-Torus."""

    def neighbours(self, x, y, z):
        yield ((x + 1) % self._dx, y, z)
        yield ((self._dx + x - 1) % self._dx, y, z)
        yield (x, (y + 1) % self._dy, z)
        yield (x, (self._dy + y - 1) % self._dy, z)
        yield (x, y, (z + 1) % self._dz)
        yield (x, y, (self._dz + z - 1) % self._dz)


class Haec(Topology):
    """Class which defines the HAEC topology."""

    def neighbours(self, x, y, z, link="both"):
        if link not in ["both", "optical", "wireless"]:
            raise ValueError("Unknown link type: '%s'" % link)
        if link != "wireless":
            # optical links
            yield ((x + 1) % self._dx, y, z)
            yield ((self._dx + x - 1) % self._dx, y, z)
            yield (x, (y + 1) % self._dy, z)
            yield (x, (self._dy + y - 1) % self._dy, z)
        if link != "optical":
            # wireless links
            if z != 0:
                for x in range(self._dx):
                    for y in range(self._dy):
                        yield (x, y, z-1)
            if z != self._dz-1:
                for x in range(self._dx):
                    for y in range(self._dy):
                        yield (x, y, z+1)
