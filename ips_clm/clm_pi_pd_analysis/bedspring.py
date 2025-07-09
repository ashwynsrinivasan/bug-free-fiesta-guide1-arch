import numpy as np
import scipy as sp
from scipy.sparse import lil_matrix, lil_array, csr_matrix, csr_array, linalg
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

rmin = 1e-6


class polygon:
    name: str
    x = None  # x-coordinates of vertices
    y = None  # y-coordinates of vertices

    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    # Check if a point is inside or on the polygon
    def is_point_inside_polygon(self, x, y):
        n = len(self.x)
        inside = False

        p1x, p1y = self.x[0], polygon.y[0]
        for i in range(n + 1):
            p2x, p2y = self.x[i % n], self.y[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    # Fill a patch and add a text annotation with it's name
    def fill_patch(self, ax, color):
        # set boundary of fill to black
        ax.fill(self.x, self.y, color, alpha=0.2, linewidth=1, edgecolor='black')
        ax.text(np.mean(self.x), np.mean(self.y), self.name, fontsize=10, horizontalalignment='center',
                verticalalignment='center')


class cell:
    x: float  # x-coordinate
    y: float  # y-coordinate
    r: int  # row-index in grid
    c: int  # column-index in grid
    n: int  # node number in system
    cx = None  # pointer to cell in the next column
    cy = None  # pointer to cell in the next row
    cz = None  # pointer to cell in the next layer
    gx: float  # conductance to the cell in the next col
    gy: float  # conductance to the cell in the next row
    gz: float = 0.0  # conductance to the cell in the next layer
    v: float  # voltage at node
    i: float  # current into node
    layer_name: str  # name of the layer that this cell belongs to

    def __init__(self, layer_name, x, y, r, c, rx, ry):
        self.layer_name = layer_name
        self.x = x
        self.y = y
        self.r = r
        self.c = c
        self.gx = 1 / rx
        self.gy = 1 / ry
        self.i = 0.0


class vsrc:  # voltage source
    global rmin
    name: str  # Name of the voltage source from which this guy is derived
    c1: cell  # pointer to cell connected to positive terminal
    c2: cell  # pointer to cell connected to negative terminal, None if ground
    gs: float  # source conductance
    n: int  # node for source resistance
    m: int  # source number
    i: float  # current out of source
    v: float  # source voltage

    def __init__(self, c1, c2, v, rs=None):
        self.c1 = c1
        self.c2 = c2
        self.v = v
        if rs is not None:
            self.gs = 1 / rs
        else:
            self.gs = 1 / rmin


class layer:  # a 2D mesh of cells
    name: str  # layer name
    net: str  # net name
    lx: float  # lower left x-coord
    ly: float  # lower left y-coord
    ux: float  # upper right x-coord
    uy: float  # upper right x-coord
    dx: float  # mesh width x
    dy: float  # mesh width y
    cells = None  # 2D mesh of cells
    xg: float
    yg: float
    nx: int
    ny: int
    rx_sheet: float
    ry_sheet: float
    custom_mesh = False

    def __init__(self, name, net, lx=0, ly=0, ux=1000, uy=1000, dx=10, dy=10, rx_sheet=10e-3, ry_sheet=10e-3,
                 custom_mesh=False, xg=None, yg=None):
        self.name = name
        self.net = net
        self.lx = lx
        self.ly = ly
        self.ux = ux
        self.uy = uy
        self.dx = dx
        self.dy = dy
        self.rx_sheet = rx_sheet
        self.ry_sheet = ry_sheet
        if not custom_mesh:
            self.generate_mesh()
        else:
            self.custom_mesh = True
            self.xg = xg
            self.yg = yg
            self.nx = xg.__len__()
            self.ny = yg.__len__()
            self.lx = xg[0]
            self.ly = yg[0]
            self.ux = xg[-1]
            self.uy = yg[-1]
            self.generate_custom_mesh()

    def generate_mesh(self):
        self.xg = np.arange(start=self.lx, stop=self.ux, step=self.dx)
        self.nx = self.xg.__len__()
        self.yg = np.arange(start=self.ly, stop=self.uy, step=self.dy)
        self.ny = self.yg.__len__()
        rx = self.rx_sheet * self.dx / self.dy
        ry = self.ry_sheet * self.dy / self.dx
        self.cells = [[cell(layer_name=self.name, x=self.xg[c], y=self.yg[r], r=r, c=c, rx=rx, ry=ry)
                       for c in range(self.nx)]
                      for r in range(self.ny)]

        # Finished generating mesh, now do the linking
        for r in range(self.ny):
            for c in range(self.nx):
                try:
                    if r < self.ny - 1:
                        self.cells[r][c].cy = self.cells[r + 1][c]
                    if c < self.nx - 1:
                        self.cells[r][c].cx = self.cells[r][c + 1]
                except:
                    print("r:" + r.__str__() + ", c:" + c.__str__())

    def generate_custom_mesh(self):
        self.cells = [[cell(layer_name=self.name, x=self.xg[c], y=self.yg[r], r=r, c=c, rx=1e6, ry=1e6)
                       for c in range(self.nx)]
                      for r in range(self.ny)]
        for r in range(self.ny):
            for c in range(self.nx):
                try:
                    if r < self.ny - 1:
                        self.cells[r][c].cy = self.cells[r + 1][c]
                        if c < self.nx - 1:
                            self.cells[r][c].ry = self.ry_sheet * (self.yg[r + 1] - self.yg[r]) / (
                                    self.xg[c + 1] - self.xg[c])
                            self.cells[r][c].gy = 1 / self.cells[r][c].ry
                    if c < self.nx - 1:
                        self.cells[r][c].cx = self.cells[r][c + 1]
                        if r < self.ny - 1:
                            self.cells[r][c].rx = self.rx_sheet * (self.xg[c + 1] - self.xg[c]) / (
                                    self.yg[r + 1] - self.yg[r])
                            self.cells[r][c].gx = 1 / self.cells[r][c].rx
                except:
                    print("r:" + r.__str__() + ", c:" + c.__str__())

    def find_closest_cell(self, x, y):
        c = np.argmin(abs(self.xg - x))
        r = np.argmin(abs(self.yg - y))
        return self.cells[r][c]

    def slice_cells(self, lx, ly, ux, uy):
        c1 = np.argmin(abs(self.xg - lx))
        c2 = np.argmin(abs(self.xg - ux))
        r1 = np.argmin(abs(self.yg - ly))
        r2 = np.argmin(abs(self.yg - uy))
        return [self.cells[r][c] for r in range(r1, r2) for c in range(c1, c2)]

    def get_cell_list(self):
        return [self.cells[r][c] for r in range(self.ny) for c in range(self.nx)]

    def plot_voltage(self):
        V = np.asarray([[self.cells[r][c].v for r in range(self.ny)] for c in range(self.nx)])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(self.xg, self.yg)
        try:
            ax.plot_surface(X, Y, V)
        except:
            ax.plot_surface(X, Y, np.transpose(V))
        # show plot
        plt.show()


class via:
    l1: layer  # Ref to bottom layer
    l2: layer  # Ref to top layer
    ls: layer  # Ref to source layer
    src_list = None  # List of voltage sources
    lx: float  # lower left x
    ly: float  # lower left y
    ux: float  # upper  right x
    uy: float  # upper right y
    dx: float  # via pitch x
    dy: float  # via pitch y
    rvia: float  # via
    xg: list = None
    yg: list = None
    isViaCurrent = False

    def plot_via_current(self, filename=None):
        if self.isViaCurrent:
            current = [s.i for s in self.src_list]
            x = [s.c1.x for s in self.src_list]
            y = [s.c1.y for s in self.src_list]
            plt.figure()
            # 3D Stem plot
            plt.scatter(x, y, c=current, cmap='jet')
            plt.xlabel('x [um]')
            plt.ylabel('y [um]')
            plt.title('Current in via')
            plt.colorbar()
            if filename is not None:
                plt.savefig(filename)
                plt.close()
            return x, y, current

    def __init__(self, l1, l2, lx=0, ly=0, ux=1000, uy=1000, dx=10, dy=10, rvia=30e-3, custom_mesh=False, xg=None,
                 yg=None, isViaCurrent=False):

        self.l1 = l1
        self.l2 = l2
        self.isViaCurrent = isViaCurrent

        if custom_mesh == False:
            if isViaCurrent:
                self.ls = layer(name="Via_{}_{}".format(l1.name, l2.name), net=l1.net,
                                lx=lx, ly=ly, ux=ux, uy=uy, dx=dx, dy=dy, rx_sheet=1e6, ry_sheet=1e6)
                self.src_list = []
            self.lx = lx
            self.ly = ly
            self.ux = ux
            self.uy = uy
            self.dx = dx
            self.dy = dy
            xg = np.arange(start=lx, stop=ux, step=dx)
            nx = xg.__len__()
            yg = np.arange(start=ly, stop=uy, step=dy)
            ny = yg.__len__()
            for c in range(nx):
                for r in range(ny):
                    c1: cell = self.l1.find_closest_cell(x=xg[c], y=yg[r])
                    c2: cell = self.l2.find_closest_cell(x=xg[c], y=yg[r])
                    if not isViaCurrent:
                        c1.cz = c2
                        c1.gz = 1 / rvia
                    else:
                        cs = self.ls.find_closest_cell(x=xg[c], y=yg[r])
                        c1.cz = cs
                        c1.gz = 1 / rvia
                        s = vsrc(c1=cs, c2=c2, v=0.0, rs=None)
                        self.src_list.append(s)

        elif custom_mesh == True:
            if isViaCurrent:
                Xg = np.unique(np.sort(xg))
                Yg = np.unique(np.sort(yg))
                self.ls = layer(name="Via_{}_{}".format(l1.name, l2.name), net=l1.net,
                                lx=np.min(xg), ly=np.min(yg), ux=np.max(xg), uy=np.max(yg), dx=dx, dy=dy, rx_sheet=1e6,
                                ry_sheet=1e6,
                                custom_mesh=True, xg=Xg, yg=Yg)
                self.src_list = []
            self.lx = np.min(xg)
            self.ly = np.min(yg)
            self.ux = np.max(xg)
            self.uy = np.max(yg)
            xg = list(xg)
            yg = list(yg)
            nx = xg.__len__()
            ny = yg.__len__()
            for ctr in range(nx):
                try:
                    c1 = self.l1.find_closest_cell(x=xg[ctr], y=yg[ctr])
                    c2 = self.l2.find_closest_cell(x=xg[ctr], y=yg[ctr])
                    if not isViaCurrent:
                        c1.cz = c2
                        c1.gz = 1 / rvia
                    else:
                        cs = self.ls.find_closest_cell(x=xg[ctr], y=yg[ctr])
                        c1.cz = cs
                        c1.gz = 1 / rvia
                        s = vsrc(c1=cs, c2=c2, v=0.0, rs=None)
                        self.src_list.append(s)
                except:
                    print("ctr:" + ctr.__str__())

        self.xg = xg
        self.yg = yg


class load:
    l1: layer  # Ref to layer load gets pulled out of
    l2: layer  # Ref to layer load gets dumped into, none if to ground
    lx: float  # lower left x
    ly: float  # lower left y
    ux: float  # upper  right x
    uy: float  # upper right y
    iload: float  # total current
    poly: polygon = None

    def __init__(self, l1, l2, lx, ly, ux, uy, iload, poly=None):
        if poly is None:
            self.l1 = l1
            self.l2 = l2
            self.lx = lx
            self.ly = ly
            self.ux = ux
            self.uy = uy
            self.iload = iload
            c1 = self.l1.slice_cells(lx, ly, ux, uy)
            for c in c1:
                c.i = -iload / c1.__len__()
            if l2 is not None:
                c2 = self.l2.slice_cells(lx, ly, ux, uy)
                for c in c2:
                    c.i = iload / c2.__len__()
            self.poly = None
        else:
            self.poly = poly
            c1 = self.l1.get_cell_list()
            # Find the cells that are inside the polygon
            c1 = [c for c in c1 if poly.is_point_inside_polygon(c.x, c.y)]
            n = c1.__len__()
            for c in c1:
                c.i = --iload / n
            if l2 is not None:
                c2 = self.l2.get_cell_list()
                c2 = [c for c in c2 if poly.is_point_inside_polygon(c.x, c.y)]
                n = c2.__len__()
                for c in c2:
                    c.i = iload / n


class load_polygon:
    filename: str = None  # TCL file containing partition boundaries
    filecontents: str = None  # TCL file contents
    loads: list = None  # List of loads

    def __init__(self, filename):
        self.filename = filename
        self.filecontents = open(filename, 'r').read()

    # Parse the TCL file and create the loads
    def parse(self):

        lines = self.filecontents.splitlines()
        # Trim to remove leading whitespaces
        lines = [l.strip() for l in lines]
        # Look for lines that start with "set group_name"
        isBlock = False
        polygons = []
        block_name = ""
        for l in lines:

            if l.startswith("set group_name"):
                block_name = l.split()[2]
                isBlock = True
                continue

            if isBlock and l.startswith("set vars($group_name,boxes)"):
                try:
                    bounds = l.split("set vars($group_name,boxes)")[1].strip().strip(
                        '"')  # Remove the "set vars($group_name,boxes)" part, trim the remaining string and remove the quotes
                    x = [float(bounds.split()[i]) for i in range(0, len(bounds.split()), 2)]
                    y = [float(bounds.split()[i]) for i in range(1, len(bounds.split()), 2)]
                except:
                    bounds = l.split('"')[1].strip()
                    x = [float(bounds.split()[i]) for i in range(0, len(bounds.split()), 2)]
                    y = [float(bounds.split()[i]) for i in range(1, len(bounds.split()), 2)]
                if x.__len__() == 2:
                    x = [[x[0], x[1], x[1], x[0], x[0]]]
                    y = [[y[0], y[0], y[1], y[1], y[0]]]
                else:
                    temp_x = x
                    temp_y = y
                    x = []
                    y = []
                    for i in range(0, temp_x.__len__(), 2):
                        x.append([temp_x[i], temp_x[i + 1], temp_x[i + 1], temp_x[i], temp_x[i]])
                        y.append([temp_y[i], temp_y[i], temp_y[i + 1], temp_y[i + 1], temp_y[i]])

                polygons.extend([polygon(block_name, x[i], y[i]) for i in range(x.__len__())])
                isBlock = False
                block_name = ""

        return polygons


class sources:
    l1: layer  # Ref to layer source +ve attaches to
    l2: layer  # Ref to layer source -ve attaches to, None if ground
    lx: float  # lower left x
    ly: float  # lower left y
    ux: float  # upper  right x
    uy: float  # upper right y
    dx: float
    dy: float
    v: float  # voltage value
    rs: float  # effective source resistance
    v_list = None  # List of voltage sources

    def __init__(self, l1, l2, lx=0, ly=0, ux=1000, uy=1000, dx=None, dy=None, v=1, rs=None, custom_mesh=False, xg=None,
                 yg=None):
        self.l1 = l1
        self.l2 = l2
        self.lx = lx
        self.ly = ly
        self.dx = dx
        self.dy = dy
        self.ux = ux
        self.uy = uy
        self.v = v
        if rs is not None:
            self.rs = rs
        else:
            self.rs = rmin

        if not custom_mesh and dx is None and dy is None:
            c1 = self.l1.slice_cells(lx, ly, ux, uy)
            if l2 is not None:
                c2 = self.l2.slice_cells(lx, ly, ux, uy)
                self.v_list = [vsrc(c1=ca, c2=cb, v=v, rs=self.rs * c1.__len__()) for ca, cb in zip(c1, c2)]
            else:
                self.v_list = [vsrc(c1=ca, c2=None, v=v, rs=self.rs * c1.__len__()) for ca in c1]
        elif not custom_mesh and dx is not None and dy is not None:
            x = np.arange(start=self.lx, stop=self.ux, step=self.dx)
            y = np.arange(start=self.ly, stop=self.uy, step=self.dy)
            xg = [x[i] for i in range(x.__len__()) for j in range(y.__len__())]
            yg = [y[j] for i in range(x.__len__()) for j in range(y.__len__())]

            self.__init__(l1=l1, l2=l2, lx=lx, ly=ly, ux=ux, uy=uy, xg=xg, yg=yg, rs=rs, v=v, custom_mesh=True)
        else:
            self.v_list = []
            for ctr in range(len(xg)):
                c1 = self.l1.find_closest_cell(x=xg[ctr], y=yg[ctr])
                if l2 is not None:
                    c2 = self.l2.find_closest_cell(x=xg[ctr], y=yg[ctr])
                    self.v_list.append(vsrc(c1=c1, c2=c2, v=v, rs=self.rs))
                else:
                    self.v_list.append(vsrc(c1=c1, c2=None, v=v, rs=self.rs))


class system:
    layers = None
    vias = None
    sources = None
    loads = None
    xls_file = None
    Y = None
    lhs = None
    rhs = None
    n_nodes = 0
    n_cells = 0
    n_vsrc = 0
    cell_list = None
    src_list = None
    dx_tsv = 20
    dy_tsv = 20
    tsv_density = 5e3  # per mm^2 (per net)

    def __init__(self):
        self.layers = []
        self.vias = []
        self.sources = []
        self.loads = []
        self.cell_list = []
        self.src_list = []
        import matplotlib
        matplotlib.use('Qt5Agg')

    def self_test2(self, dx_tsv=100, dy_tsv=100, tsv_density=1111):
        self.layers = []
        self.vias = []
        self.sources = []
        self.loads = []
        ux = 1500
        uy = 1500
        dx = 20
        dy = 20
        self.dx_tsv = dx_tsv
        self.dy_tsv = dy_tsv

        l1 = layer(name="Package_Vdd0p85", net="vcc", lx=0, ly=0, ux=ux, uy=uy, dx=dx, dy=dy,
                   rx_sheet=2.65e-3, ry_sheet=2.65e-3)
        l2 = layer(name="Base_Die RDL", net="vcc", lx=0, ly=0, ux=ux, uy=uy, dx=dx, dy=dy,
                   rx_sheet=80e-3, ry_sheet=1e3)
        l3 = layer(name="Base_Die Top", net="vcc", lx=0, ly=0, ux=ux, uy=uy, dx=dx, dy=dy,
                   rx_sheet=30.8e-3, ry_sheet=60e-3)
        l4 = layer(name="Top_Die Top", net="vcc", lx=0, ly=0, ux=ux, uy=uy, dx=dx, dy=dy,
                   rx_sheet=30.8e-3, ry_sheet=60e-3)

        l5 = layer(name="Top die_Vss", net="vss", lx=0, ly=0, ux=ux, uy=uy, dx=dx, dy=dy,
                   rx_sheet=30.8e-3, ry_sheet=60e-3)
        l6 = layer(name="Base die_Vss", net="vss", lx=0, ly=0, ux=ux, uy=uy, dx=dx, dy=dy,
                   rx_sheet=80e-3, ry_sheet=30.8e-3)
        l7 = layer(name="Base die_RDL Vss", net="vss", lx=0, ly=0, ux=ux, uy=uy, dx=dx, dy=dy,
                   rx_sheet=80e-3, ry_sheet=1e3)
        l8 = layer(name="Package_Vss", net="vss", lx=0, ly=0, ux=ux, uy=uy, dx=dx, dy=dy,
                   rx_sheet=2.65e-3, ry_sheet=2.65e-3)

        via_count = dx_tsv * dy_tsv * tsv_density / 1e6
        rvia = 100e-3 / via_count

        v1 = via(l1=l1, l2=l2, lx=75, ly=75, ux=ux - 75, uy=uy - 75, dx=150, dy=334.8, rvia=1e-3)
        v2 = via(l1=l2, l2=l3, lx=dx_tsv, ly=dy_tsv, ux=ux - 75, uy=uy - 75, dx=dx_tsv, dy=dy_tsv, rvia=rvia)
        v3 = via(l1=l4, l2=l5, lx=dx_tsv, ly=dy_tsv, ux=ux - 75, uy=uy - 75, dx=dx_tsv, dy=dy_tsv, rvia=rvia)
        v4 = via(l1=l5, l2=l6, lx=150, ly=204.9, ux=ux - 75, uy=uy - 75, dx=150, dy=334.8, rvia=1e-3)

        s1 = sources(l1=l2, l2=None, lx=0, ly=0, ux=ux, uy=50, v=1.0, rs=10e-3)
        s2 = sources(l1=l6, l2=None, lx=0, ly=60, ux=ux, uy=uy, v=0.0, rs=3.84e-3)
        ld = load(l3, l4, 100, 100, ux - 100, uy - 100, 10)
        self.layers.extend([l1, l2, l3, l4, l5, l6])
        self.vias.extend([v1, v2, v3, v4])
        self.sources.extend([s1, s2])
        self.loads.append(ld)
        self.arrange_nodes()
        self.fill_matrix()
        self.solve()
        self.recover()
        self.plot_voltage(l3, 'voltage_distribution_{}'.format(l3.name))
        plt.show()
        # self.plot_current()

    def stamp(self, g, n1, n2=None):
        self.Y[n1, n1] = self.Y[n1, n1] + g

        if n2 is not None:
            self.Y[n1, n2] = self.Y[n1, n2] - g
            self.Y[n2, n1] = self.Y[n2, n1] - g
            self.Y[n2, n2] = self.Y[n2, n2] + g

    def stamp_src(self, v):
        self.Y[v.m, v.n] = 1
        self.Y[v.n, v.m] = 1
        self.rhs[v.m, 0] = v.v
        self.stamp(g=v.gs, n1=v.c1.n, n2=v.n)
        if v.c2 is not None:
            self.Y[v.m, v.c2.n] = -1
            self.Y[v.c2.n, v.m] = -1

    def arrange_nodes(self):
        n = 0
        m = 0
        self.cell_list = []
        self.src_list = []
        v: via
        for v in self.vias:
            if v.isViaCurrent:
                self.layers.append(v.ls)

        # number the cell nodes first
        for l in self.layers:
            for c in l.get_cell_list():
                c.n = n
                n = n + 1
            self.cell_list.extend(l.get_cell_list())
        self.n_cells = n

        # number the voltage sources next
        for s in self.sources:
            for src in s.v_list:
                src.n = n
                n = n + 1
        for v in self.vias:
            if v.isViaCurrent:
                for src in v.src_list:
                    src.n = n
                    n = n + 1

        for s in self.sources:
            for src in s.v_list:
                src.m = m + n
                m = m + 1
            self.src_list.extend(s.v_list)
        for v in self.vias:
            if v.isViaCurrent:
                for src in v.src_list:
                    src.m = m + n
                    m = m + 1
                self.src_list.extend(v.src_list)

        self.n_nodes = n
        self.n_vsrc = m

    def fill_matrix(self):
        n = self.n_nodes
        m = self.n_vsrc
        k = self.n_cells

        self.Y = lil_matrix((n + m, n + m))
        self.rhs = lil_array((n + m, 1))

        for c in self.cell_list:
            if c.cx is not None:
                self.stamp(g=c.gx, n1=c.n, n2=c.cx.n)
            if c.cy is not None:
                self.stamp(g=c.gy, n1=c.n, n2=c.cy.n)
            if c.cz is not None:
                try:
                    self.stamp(g=c.gz, n1=c.n, n2=c.cz.n)
                except:
                    print(c)
            try:
                self.rhs[c.n, 0] = c.i
            except:
                print(c)

        for v in self.src_list:
            self.stamp_src(v)

    def solve(self):
        Y = self.Y.tocsr()
        rhs = self.rhs.tocsr()
        self.lhs = linalg.spsolve(Y, rhs)

    def recover(self):
        n = 0
        for c in self.cell_list:
            c.v = self.lhs[n]
            n = n + 1
        m = 0
        for s in self.src_list:
            s.i = self.lhs[self.n_nodes + m]
            m = m + 1

    # Function to plot voltage on a layer
    def plot_voltage(self, l, xp=None, yp=None, filename=None):
        plt.figure()
        x = []
        y = []
        v = []

        if xp is None:
            for c in l.get_cell_list():
                x.append(c.x)
                y.append(c.y)
                v.append(c.v)
        else:
            x = xp
            y = yp
            v = [l.find_closest_cell(x=xp[i], y=yp[i]).v for i in range(xp.__len__())]
        plt.scatter(x, y, c=v, cmap='jet')
        plt.colorbar()
        plt.xlabel('X-dimension (um)')
        plt.ylabel('Y-dimension (um)')
        plt.title('Voltage distribution on layer {} & Net {}'.format(l.name, l.net))
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
        return x, y, v

    # Function to plot current from voltage sources
    def plot_current(self):
        plt.figure()
        x = []
        y = []
        i = []
        for v in self.src_list:
            if (not v.i == 0) and (v.v == 0):
                x.append(v.c1.x)
                y.append(v.c1.y)
                i.append(np.abs(v.i))
        plt.scatter(x, y, c=i, cmap='jet')
        plt.colorbar()
        plt.show()

    # Function to plot locations of x,y coordinates for each layer
    def plot_layer(self):
        plt.figure()
        for l in self.layers:
            x = []
            y = []
            for c in l.get_cell_list():
                x.append(c.x)
                y.append(c.y)
            plt.scatter(x, y, c='black')
        plt.show()


if __name__ == '__main__()':
    matplotlib.use('Qt5Agg')
    plt.figure()
    plt.show()
    s = system()
    s.self_test2()
