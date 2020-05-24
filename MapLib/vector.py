from math import floor, ceil, sqrt


class Vector():
    """Class wich represents a vector and defines operations on it."""

    def __init__(self, x, y, z):
        self.value = x, y, z

    def __iter__(self):
        return self.value.__iter__()

    @property
    def norm(self):
        """
        Returns the lenght of a vector.
        :return: Vector length.
        """
        return sqrt(sum([s*s for s in self]))

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.value == other.value

    def __neg__(self):
        return Vector(*[-s for s in self])

    def __add__(self, other):
        if type(other) == type(self):
            return Vector(*[s+o for s, o in zip(self, other)])
        else:
            raise ValueError("Not defined for %s and %s" % (self, other))

    def __sub__(self, other):
        if type(other) == type(self):
            return self.__add__(other.__neg__())
        else:
            raise ValueError("Not defined for %s and %s" % (self, other))

    def __mul__(self, other):
        if type(other) == type(self):
            return Vector(*[s*o for s, o in zip(self, other)])
        elif type(other) == type(1):
            return Vector(*[s*other for s in self])
        else:
            raise ValueError("Not defined for %s and %s" % (self, other))

    def __floordiv__(self, other):
        if type(other) == type(self):
            return Vector(*[floor(s/o) if s*o > 0 else ceil(s/o) for s, o in zip(self, other)])
        elif type(other) == type(1) and other != 0:
            return Vector(*[floor(s/other) if s*other > 0 else ceil(s/other) for s in self])
        else:
            raise ValueError("Not defined for %s and %s" % (self, other))

    def __repr__(self):
        return str(self.value)
