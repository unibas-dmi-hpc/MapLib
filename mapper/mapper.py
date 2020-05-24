import click
import logging
import logging.config
from functools import update_wrapper
from os.path import abspath, dirname, join

from . mapping import Sweep, Scan, Peano, Gray, Hilbert, Diagonal
from . mapping import Bokhari, MinimumManhattanDistance, Topology_aware
from . mapping import Recursive_bipartitioning, Greedy_Graph_Embedding, Greedy_All_C, UDFS
from . mapping import Genetic, Random
from . mapping_compl import pacMAP, Fast_High_Greedy, A_star, A_star_Ali, One_to_One


from .topology import Topology
from .graph import Graph
from .visualization import sketch_to_pdf
from .routing import route

from .statistics import ieplc, ianlc, ienlc, ienpc, wtad, wtad_default, wtad_al, NA, LA, AD, DFC, Diam

logging_conf = join(abspath(dirname(__file__)), 'logging_conf.ini')
logging.config.fileConfig(logging_conf)
logger = logging.getLogger(__name__)
mappings = {'sweep': Sweep,
            'diagonal': Diagonal,
            'topo_aware': Topology_aware,
            'bipartition': Recursive_bipartitioning,
            'greedy': Greedy_Graph_Embedding,
            'greedyALLC': Greedy_All_C,
            'udfs': UDFS,
            'pacmap': pacMAP,
            'FHGreedy':Fast_High_Greedy,
            'one_to_one':One_to_One,
            'scan': Scan,
            'peano': Peano,
            'gray': Gray,
            'hilbert': Hilbert,
            'min-md': MinimumManhattanDistance,
            'astar': A_star,
            'ali_astar' : A_star_Ali,
            'bokhari': Bokhari,
            'genetic': Genetic,
            'random': Random
            }

"""Base module for the command-line application."""


@click.group(chain=True)
@click.option('-v', '--verbose', 'verbosity', count=True,
              help='Increase output verbosity.')
def cli(verbosity):
    pass

@cli.resultcallback()
def process_commands(processors, verbosity):
    """
    Callback method to process input stream
    Was taken from click (click.pocoo.com) examples.
    :param processors: Processors
    :param verbosity: Verbosity
    :return:
    """
    logger.info("Start...")
    stream = ()
    try:
        for processor in processors:
            stream = processor(stream)

        for _ in stream:
            pass
    except KeyboardInterrupt:
        logger.critical("Manually interrupted execution!")
    except Exception as e:
        logger.exception(e)


def processor(f):
    """ Was taken from click (click.pocoo.com) examples."""
    def new_func(*args, **kwargs):
        def processor(stream):
            return f(stream, *args, **kwargs)
        return processor
    return update_wrapper(new_func, f)


def generator(f):
    """ Was taken from click (click.pocoo.com) examples."""
    @processor
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in f(*args, **kwargs):
            yield item
    return update_wrapper(new_func, f)


@cli.command('map')
@click.option('-i', '--input', 'fname', type=click.File('r'), multiple=True,
              help='Communication matrix file.')
@click.option('-I', '--inputs', 'fnames', type=click.File('r'), nargs=2, multiple=True,
              help='Communication matrix file.')
@click.option('-m', '--mapping', 'mname', type=str, default='sweep',
              help='Name of the mapping method.')
@click.option('-t', '--topology', 'tname', type=str, default='mesh',
              help='Name of the topology.')
@click.option('-d', '--dim', 'dim', type=(int, int, int),
              default=(4, 4, 4), help='Dimensions of the topology, default: 4x4x4')
@click.option('-s', '--delimiter', type=str, default=',',
              help='Value delimiter in the communication matrix file, default: \',\'')
@click.option('-b', '--block', is_flag=True, default=False,
              help='')
@generator
def map_cmd(fname, fnames, mname, tname, dim, delimiter, block):
    """Generates mapping from communication matrix."""
    info_str = ('MAP - file name: "%s" mapping name: "%s" '
                'topology name: "%s" dimensions: "%s"')
    for f in fname + fnames:
        logger.info(info_str % (f, mname, tname, dim))
        process_graph = Graph(f, delimiter)
        topology = Topology.get_topology(tname, dim)
        Mapping = mappings[mname]
        yield Mapping(process_graph, topology, logger=logger, block=block)


@cli.command('route')
@click.option('-a', '--algorithm', 'routing_alg', type=str, default='xyz',
              help='Name of routing algorihtm.')
@processor
def route_cmd(mappings, routing_alg):
    """Routes messages in the topology."""
    for mapping in mappings:
        mapping.routed = True
        yield mapping


@cli.command('stat')
@processor
def stat_cmd(mappings):
    """Returns the inter/intra node/process statistics."""
    for mapping in mappings:
        print("(total, avg, min, max)")
        print("IePLC  :", ieplc(mapping))
        print("IaNLC  :", ianlc(mapping))
        print("IeNLC  :", ienlc(mapping))
        print("IeNPC  :", ienpc(mapping))
        print("WTAD-default :",wtad_default(mapping))
        print("WTAD   :", wtad(mapping))
        print("WTAD_AL:", wtad_al(mapping))
        print("NA     :", NA(mapping))
        print("LA     :", LA(mapping))
        print("AD     :", AD(mapping))
        print("DFC    :", DFC(mapping))
        print("Diam   :", Diam(mapping))
        yield mapping


@cli.command('vis')
@click.option('-o', '--output', 'fname', type=str,
              required=True, help='Name of the generated pdf file')
@click.option('-d', '--display', is_flag=True, default=False,
              help='')
@click.option('-n', '--non-neighbors', is_flag=True, default=False,
              help='')
@processor
def vis_cmd(mappings, fname, display, non_neighbors):
    """Visualizes a mapping."""
    for mapping in mappings:
        fname = sketch_to_pdf(mapping, fname, non_neighbors)
        if display:
            click.launch(fname)
        yield mapping

@cli.command('save')
@click.option('-o', '--output', 'fname', type=click.File('w'),
              help='Name of the generated pdf file')
@click.option('-d', '--display', is_flag=True, default=False,
              help='')
@processor
def save_cmd(mappings, fname, display):
    """Saves mapping to file."""
    for mapping in mappings:
        fname.write(str(mapping))
        if display:
            click.launch(fname)
        yield mapping

@cli.command('matrix')
@processor
def matrix_cmd(mappings):
    """Generates minimal hop matrix"""
    for mapping in mappings:
        num_procs = len(mapping.mapping)
        for u in range(num_procs):
            line = []
            for v in range(num_procs):
                if not mapping.process_graph.has_edge(u,v):
                    line.append(0)
                else:
                    line.append(mapping.topology.hops(mapping.mapping[u],
                                                      mapping.mapping[v]))
            print(','.join([str(i) for i in line]))

        yield mapping


@cli.command('print')
@processor
def print_cmd(mappings):
    """Prints mapping to stdout"""
    for mapping in mappings:
        click.echo(mapping)
        yield mapping
