from os import getcwd, path
from os.path import exists, join
from shutil import move
from subprocess import Popen, TimeoutExpired, PIPE
import re
import tempfile

from .routing import route, get_edges, group_edges


"""Module for visualization of mappings"""


SKETCH_TEMPLATE = r"""
def dx %d
def dy %d
def dz %d

def offset -0.8
put {scale(0.5) * translate([offset*2, offset/2, offset/2])}{
    def l 0.7
    def o 0.4
    line[arrows=->] (0,0,0)(l,0,0)
    line[arrows=->] (0,0,0)(0,l,0)
    line[arrows=->] (0,0,0)(0,0,l)
    special |\path #1 node {\scriptsize $x$}
                   #2 node {\scriptsize $y$}
                   #3 node {\scriptsize $z$};| (l+o,0,0)(0,l+o,0)(0,0,l+o)
}

repeat {dz, translate([0,0,1])}{
    repeat {dx, translate([1,0,0])}{
        line[style=ultra thin, lay=under, color=colorg] (0,0,0)(0,dy-1,0)
    }
}
repeat {dy, translate([0,1,0])}{
    repeat {dz, translate([0,0,1])}{
        line[style=ultra thin, lay=under, color=colorg] (0,0,0)(dx-1,0,0)
    }
}
repeat {dx, translate([1,0,0])}{
    repeat {dy, translate([0,1,0])}{
        line[style=ultra thin, lay=under, color=colorg] (0,0,0)(0,0,dz-1)
    }
}

%s

global {
    language tikz
    camera rotate(0, (1,0,0)) *
    view((-2.5,2,1.5),(0,0,0),[0,0,1]) *
    rotate(180, (0,0,1))
}
"""

TEX_TEMPLATE = r"""
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usepackage{tikz-3dplot}
\usetikzlibrary{patterns}

\definecolor{colorg}{HTML}{D3D3D3}
\definecolor{color0}{HTML}{000000}
\definecolor{color1}{HTML}{3288BD}
\definecolor{color2}{HTML}{66C2A5}
\definecolor{color3}{HTML}{ABDDA4}
\definecolor{color4}{HTML}{E6F598}
\definecolor{color5}{HTML}{FEE08B}
\definecolor{color6}{HTML}{FDAE61}
\definecolor{color7}{HTML}{F46D43}
\definecolor{color8}{HTML}{D53E4F}


\begin{document}

%%SKETCH_OUTPUT%%

\end{document}"""



def sketch_to_tex(sketch_input):
    """
    Calls sketch to process the sketch input to tex
    :param sketch_input: Sketch input file
    :return: Tex string
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(TEX_TEMPLATE.encode())
    try:
        proc = Popen(["sketch", "-t", temp_file.name], stdin=PIPE, stdout=PIPE,
                     stderr=PIPE)
        out, err = proc.communicate(input=sketch_input.encode(), timeout=15)
        return out.decode()
    except TimeoutExpired:
        proc.terminate()
        _, err = proc.communicate()
        raise RuntimeError("Sketch not responding.")


def get_filename(path):
    """
    Checks if a given filename is already present at the location and returns
    the name with an numbered suffix if so.
    :param path: Output path
    :return: File path
    """
    counter = 1
    try:
        name, extension = path.rsplit('.', 1)
    except ValueError:
        name, extension = path, 'pdf'
        path += '.pdf'
    while exists(path):
        path = (name + '(%d).' + extension) % counter
        counter += 1
    return path


def tex_to_pdf(tex_input, file_name):
    """
    Calls pdflatex to convert tex file into pdf file.
    :param tex_input: Input tex file
    :param file_name: Name of the pdf file to be generated
    :return: Name of the generated pdf file
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Replace file extension only if is equal to 'pdf'
            tex_file = path.join(temp_dir, re.sub(r'.pdf$', '', file_name) + '.tex')
            with open(tex_file, 'w+') as f:
                f.write(tex_input)
            proc = Popen(["pdflatex", tex_file], cwd=temp_dir, stdout=PIPE,
                         stderr=PIPE)
            out, err = proc.communicate(timeout=15)
            outfile = get_filename(join(getcwd(), file_name))
            move(re.sub(r'.tex', '.pdf', tex_file), outfile)
            return outfile
        except TimeoutExpired as e:
            proc.terminate()
            out, _ = proc.communicate()
            raise RuntimeError("PDFLatex not responding.")


def calc_color_value(value, _min, _max):
    """
    Calculates a color value based on minimum and maximum value.
    :param value: Value
    :param _min: Lower value bound
    :param _max: Higher value bound
    :return:
    """
    num_colors = 8
    black = 0
    if _min == _max:
        return black
    return round((num_colors-1) * ((value - _min) / (_max - _min))) + 1


def create_sketch_input(mapping):
    """
    Calculates the sketch input file for a given mapping.
    :param mapping: Process to node mapping
    :return: The sketch input file
    """
    line_template = "line[style=very thin, draw=color%d] %s%s "
    lines = []
    if mapping.routed:
        edges = group_edges(route(mapping))
    else:
        edges = group_edges(get_edges(mapping))
    _min, _max = min([w for u, v, w in edges] or [0]), max([w for u, v, w in edges] or [0])
    for u, v, w in edges:
        color = calc_color_value(w, _min, _max)
        lines.append(line_template % (color, u, v))
    dx, dy, dz = mapping.topology.dim
    return SKETCH_TEMPLATE % (dx, dy, dz, "\n".join(lines))


def sketch_to_pdf(mapping, file_name, none_neighbors):
    sketch_input = create_sketch_input(mapping, none_neighbors)
    sketch_output = sketch_to_tex(sketch_input)
    return tex_to_pdf(sketch_output, file_name)
