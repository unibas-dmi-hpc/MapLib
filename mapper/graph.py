from csv import reader
import networkx as nx
from click import open_file

class Graph(nx.Graph):
    """Class for a communication graph."""

    def from_single_file(self, fdata, sep):
        """
        Returns a graph from a communication matrix
        :param fdata: Csv file
        :param sep: Value delimiter
        :return: Communication Graph
        """
        with open(fdata.name) as f:
            for u, line in enumerate(reader(f, delimiter=sep)):
                for v, val in enumerate(line):
                    val = int(val)
                    if val == 0:
                        continue
                    if self.has_edge(u, v):
                        val += self.get_edge_data(u, v)['weight']
                    self.add_edge(u, v, weight=val)

    def from_two_files(self, fcount, fsize, sep):
        """
        Returns a graph from a communication matrix (count and size)
        :param fcount: Count communication csv file
        :param fsize: Size communication csv file
        :param sep: Value delimiter
        :return: Communication Graph
        """
        with open(fcount.name) as f1, open(fsize.name) as f2:
            for u, (l1, l2) in enumerate(zip(reader(f1, delimiter=sep), reader(f2, delimiter=sep))):
                for v, (val1, val2) in enumerate(zip(l1, l2)):
                    val1, val2 = int(val1), int(val2)
                    if val1 == 0 and val2 == 0:
                        continue
                    if self.has_edge(u, v):
                        val1 += self.get_edge_data(u, v)['count']
                        val2 += self.get_edge_data(u, v)['size']
                    self.add_edge(u, v, count=val1, size=val2)

    def __init__(self, ifile, delimiter):
        super().__init__()
        if type(ifile) != tuple:
            self.from_single_file(ifile, delimiter)
        else:
            fcount, fsize = ifile
            self.from_two_files(fcount, fsize, delimiter)
