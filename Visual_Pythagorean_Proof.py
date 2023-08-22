from manimlib.imports import *
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt


class Grid(VGroup):
    CONFIG = {
        "height": 8.0,
        "width": 8.0,
    }

    def __init__(self, rows, columns, **kwargs):
        digest_config(self, kwargs, locals())
        super().__init__(**kwargs)

        x_step = self.width / self.columns
        y_step = self.height / self.rows

        for x in np.arange(0, self.width + x_step, x_step):
            self.add(Line(
                [x - self.width / 2., -self.height / 2., 0],
                [x - self.width / 2., self.height / 2., 0],
            ))
        for y in np.arange(0, self.height + y_step, y_step):
            self.add(Line(
                [-self.width / 2., y - self.height / 2., 0],
                [self.width / 2., y - self.height / 2., 0]
            ))


class ScreenGrid(VGroup):
    CONFIG = {
        "rows": 20,
        "columns": 20,
        "height": FRAME_Y_RADIUS * 2,
        "width": 14,
        "grid_stroke": 0.5,
        "grid_color": WHITE,
        "axis_color": RED,
        "axis_stroke": 2,
        #"labels_scale": 0.25,
        #"labels_buff": 0,
        "number_decimals": 2
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rows = self.rows
        columns = self.columns
        grid = Grid(width=self.width, height=self.height, rows=rows, columns=columns)
        grid.set_stroke(self.grid_color, self.grid_stroke)

        vector_ii = ORIGIN + np.array((- self.width / 2, - self.height / 2, 0))
        vector_si = ORIGIN + np.array((- self.width / 2, self.height / 2, 0))
        vector_sd = ORIGIN + np.array((self.width / 2, self.height / 2, 0))

        axes_x = Line(LEFT * self.width / 2, RIGHT * self.width / 2)
        axes_y = Line(DOWN * self.height / 2, UP * self.height / 2)

        axes = VGroup(axes_x, axes_y).set_stroke(self.axis_color, self.axis_stroke)

        divisions_x = self.width / columns
        divisions_y = self.height / rows

        directions_buff_x = [UP, DOWN]
        directions_buff_y = [RIGHT, LEFT]
        dd_buff = [directions_buff_x, directions_buff_y]
        vectors_init_x = [vector_ii, vector_si]
        vectors_init_y = [vector_si, vector_sd]
        vectors_init = [vectors_init_x, vectors_init_y]
        divisions = [divisions_x, divisions_y]
        orientations = [RIGHT, DOWN]
        #labels = VGroup()
        set_changes = zip([columns, rows], divisions, orientations, [0, 1], vectors_init, dd_buff)
        for c_and_r, division, orientation, coord, vi_c, d_buff in set_changes:
            for i in range(1, c_and_r):
                for v_i, directions_buff in zip(vi_c, d_buff):
                    ubication = v_i + orientation * division * i
                    coord_point = round(ubication[coord], self.number_decimals)
                    #label = Text(f"{coord_point}",font="Arial",stroke_width=0).scale(self.labels_scale)
                    #label.next_to(ubication, directions_buff, buff=self.labels_buff)
                    #labels.add(label)

        self.add(grid, axes)#, labels)




'''THIS IS A CODE USED TO ANIMATE PLOTS'''

class GraphFromData(GraphScene):
    # Covert the data coords to the graph points
    def get_points_from_coords(self,coords):
        return [
            # Convert COORDS -> POINTS
            self.coords_to_point(px,py)
            # See manimlib/scene/graph_scene.py
            for px,py in coords
        ]

    # Return the dots of a set of points
    def get_dots_from_coords(self,coords,radius=0.1):
        points = self.get_points_from_coords(coords)
        dots = VGroup(*[
            Dot(radius=radius).move_to([px,py,pz])
            for px,py,pz in points
            ]
        )
        return dots

class DiscreteGraphFromSetPoints(VMobject):
    def __init__(self,set_of_points,**kwargs):
        super().__init__(**kwargs)
        self.set_points_as_corners(set_of_points)

class SmoothGraphFromSetPoints(VMobject):
    def __init__(self,set_of_points,**kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly(set_of_points)
'''INPUT LISTS AND PROPERTIES BELOW'''
class Uniqueplot(GraphFromData):
    CONFIG = {
        "y_max": 5,
        "x_max": 10,
    }
    def construct(self):
        self.setup_axes(animate=True)
        x = [x for x in np.linspace(0,10,5)]
        y = [y for y in range(0,10)]

        coords = [[px,py] for px,py in zip(x,y)]
        # |
        # V
        points = self.get_points_from_coords(coords)
        
        graph = SmoothGraphFromSetPoints(points,color=GREEN)
        dots = self.get_dots_from_coords(coords)

        self.play(ShowCreation(graph),ShowCreation(dots))


class Uniqueplot22(GraphFromData):
    CONFIG = {
        "y_max": 100,
        "x_max": 10,
        "y_tick_frequency" : 10
    }
    def construct(self):
        grid = ScreenGrid()
        self.setup_axes(animate=True)
        
        def f(x):
            y = x**2 + 2
            return y

        xlist = []
        ylist = []
        for i in np.linspace(0,8,1000):
            xlist.append(i)
            ylist.append(f(i))

        coords = [[px,py] for px,py in zip(xlist,ylist)]
        # |
        # V
        points = self.get_points_from_coords(coords)
        
        graph = SmoothGraphFromSetPoints(points,color=GREEN)
        #dots = self.get_dots_from_coords(coords)

        dot = Dot().move_to([0,0,0])
        self.play(ShowCreation(graph))#,ShowCreation(dots), Write(dot))
        self.play(ApplyMethod(dot.shift,RIGHT*2))
        self.add(grid)
        self.wait()
        

class Plotnodots(GraphFromData):
    CONFIG = {
        "y_max" : 100,
        "y_min" : 0,
        "x_max" : 300,
        "x_min" : 0,
        "y_tick_frequency" : 10**21, 
        "x_tick_frequency" : 60, 
        "axes_color" : BLUE,
        "y_axis_label" : r"$\Sigma \cdot 10^{20}$ [kg/m$^3$]",
        "graph_origin": 3.5 * DOWN + 5 * LEFT,
        "y_labeled_nums" : list(np.arange(10**2, 10**3, 100))
    }
    def construct(self):
        self.setup_axes(animate=True)
        def column_dens(r):
            E = (6.30*10**15/((3*np.pi*0.01*(9.9*10**2*(2.34/2.35*r**(-1/2))**1/2)**2)/np.sqrt((6.67*10**(-11)*1.989*10**30)/r**3)))/10**20
            return E

        xlist = []
        ylist = []
        for i in np.linspace(0.1,300,1000):
            xlist.append(i)
            ylist.append(column_dens(i))

        coords = [[px,py] for px,py in zip(xlist,ylist)]
        # |
        # V
        points = self.get_points_from_coords(coords)
        
        graph = SmoothGraphFromSetPoints(points,color=GREEN)
        #dots = self.get_dots_from_coords(coords)

        self.play(ShowCreation(graph))
        self.wait()


class orbits(GraphFromData):
    CONFIG = {
        "y_max": 8,
        "x_max": 8,
        "y_tick_frequency" : 10,
        "y_min": -8,
        "x_min": -8,
        "graph_origin": 0
    }
    def construct(self):
        #grid = ScreenGrid()
        self.setup_axes(animate=False)
        
        particles = 4
        masslist = [1*u.solMass, 0.001*u.solMass, 0*u.solMass, 0*u.solMass] #Solar masses
        w_0 = (0*u.AU, 0*u.AU, 0*u.AU, 0*u.AU/u.year, 0*u.AU/u.year, 0*u.AU/u.year) #Au, Au/år
        w_1 = (0*u.AU, 5.2*u.AU, 0*u.AU, -2.75674*u.AU/u.year, 0*u.AU/u.year, 0*u.AU/u.year)
        w_2 = (-4.503*u.AU, 2.6*u.AU, 0*u.AU, -1.38*u.AU/u.year, -2.39*u.AU/u.year, 0*u.AU/u.year)
        w_3 = (4.503*u.AU, 2.6*u.AU, 0*u.AU,  -1.38*u.AU/u.year, 2.39*u.AU/u.year, 0*u.AU/u.year)

        'Constants'
        #G = 6.67408 * 10**(-11)
        G = 4*np.pi**2*((u.AU**3)/(u.solMass*u.year**2))
        h = 0.01*u.year #år
        period = 20*u.year
        steps = int((period)/h)

        # Arrays for values
        xpos = np.zeros((particles, steps))
        ypos = np.zeros((particles, steps))
        zpos = np.zeros((particles, steps))
        velx = np.zeros((particles, steps))
        vely = np.zeros((particles, steps))
        velz = np.zeros((particles, steps))

        W = []
        W.append(w_0)
        W.append(w_1)
        W.append(w_2)
        W.append(w_3)
            
        # initial positions
        for i in range(particles): 
            xpos[i][0] = W[i][0].value
            ypos[i][0] = W[i][1].value
            zpos[i][0] = W[i][2].value
            velx[i][0] = W[i][3].value
            vely[i][0] = W[i][4].value
            velz[i][0] = W[i][5].value

        for k in range(1,steps): #skipping first time step

            for i in range(1, particles): #Skipping the sun
                wi = W[i] #state vector
                vxi = wi[3]
                vyi = wi[4]
                vzi = wi[5]
                
                xi = wi[0]
                yi = wi[1]
                zi = wi[2]
                
                Forcelist = np.zeros(len(W)) * u.AU/(u.year)**2
                for j in range(particles):
                    if j != i:
                        wj = W[j] 
                        
                        xj = wj[0]
                        yj = wj[1]
                        zj = wj[2]
                        
                        rad = (xi-xj, yi-yj, zi-zj)
                        absrad = ((rad[0])**2 + rad[1]**2 + rad[2]**2)**(0.5)
                        
                        dvxi = -G * ((masslist[j])*(xi-xj))/((absrad)**3)
                        dvyi = -G * ((masslist[j])*(yi-yj))/((absrad)**3)
                        dvzi = -G * ((masslist[j])*(zi-zj))/((absrad**3))
                        
                        Forcelist[0] = Forcelist[0] + dvxi
                        Forcelist[1] = Forcelist[1] + dvyi
                        Forcelist[2] = Forcelist[2] + dvzi

                vxi = vxi + Forcelist[0]*h
                vyi = vyi + Forcelist[1]*h
                vzi = vzi + Forcelist[2]*h
                        
                xi = xi + vxi*h
                yi = yi + vyi*h
                zi = zi + vzi*h
                        
                wi = (xi, yi, zi, vxi, vyi, vzi)
                W[i] = wi

                xpos[i][k] = xi.value
                ypos[i][k] = yi.value
                zpos[i][k] = zi.value
                
                velx[i][k] = vxi.value
                vely[i][k] = vyi.value
                velz[i][k] = vzi.value
        
        # points for the bodies
        jupiter = [[px,py] for px,py in zip(xpos[1],ypos[1])] 
        troj1 = [[px,py] for px,py in zip(xpos[2], ypos[2])]
        troj2 = [[px,py] for px,py in zip(xpos[3], ypos[3])]

        # Apply pre-defined class to obtain coordinates from points
        points = self.get_points_from_coords(jupiter)
        points2  = self.get_points_from_coords(troj1)
        points3 = self.get_points_from_coords(troj2)
        
        # Define graph and animation properties from class
        graph = SmoothGraphFromSetPoints(points,color=GREEN)
        graph2 = SmoothGraphFromSetPoints(points2, color=RED)
        graph3 = SmoothGraphFromSetPoints(points3, color = YELLOW)

        p = Dot()
        p2 = p.copy()
        p3 = p.copy()

        dot = Circle(color = YELLOW).move_to([xpos[0][0],ypos[0][0],0])
        text = TextMobject('Sun')
        text.scale(0.6)
        text.next_to(dot, buff = TOP)
        jup = TextMobject('Jupiter')
        jup.scale(0.6)
        jup.next_to(p, buff = LEFT)
        troj = TextMobject('Trojan 1')
        troj.scale(0.6)
        troj.next_to(p2, buff = LEFT)
        troj2 = TextMobject('Trojan 2')
        troj2.scale(0.6)
        troj2.next_to(p3, buff = LEFT)
        self.play(Write(text), Write(dot))
        self.play(MoveAlongPath(p,graph, rate_func = linear), MoveAlongPath(p2, graph2, rate_func = linear),MoveAlongPath(p3, graph3, rate_func = linear), MoveAlongPath(jup,p),MoveAlongPath(troj,p2),MoveAlongPath(troj2,p3), run_time = 15)
        self.wait()


class neworbits(GraphFromData):

    CONFIG = {
        "y_max": 8,
        "x_max": 8,
        "y_tick_frequency" : 2,
        "x_tick_frequency" : 2,
        "y_min": -8,
        "x_min": -8,
        "graph_origin": 0,
        "y_labeled_nums": list(np.arange(0,8,2)),
        "x_labeled_nums": list(np.arange(0,8,2)),
    }
    def construct(self):
        #grid = ScreenGrid()
        self.setup_axes(animate=True)

        import numpy as np

        names = ['Jupiter', 'Trojan 1','Trojan 2']

        w_0 = [0, 0, 0, 0, 0, 0] #Au, Au/år
        w_1 = [0, 5.2, 0, -2.75674, 0, 0]
        w_2 = [-4.503, 2.6, 0, -1.38, -2.39, 0]
        w_3 = [4.503, 2.6, 0,  -1.38, 2.39, 0]
        W = np.array([w_0, w_1, w_2, w_3])

        Mass = [1, 0.001, 0, 0]
        particles = 4
        G = 4*np.pi**2
        def Force(xi, xj, mass_j):
            rad = [(xi[i] - xj[i]) for i in range(3)]
            absrad = ((rad[0])**2 + (rad[1])**2 + (rad[2])**2)**(0.5)
            f = [-G * ((mass_j)*(rad[i]))/((absrad)**3) for i in range(3)]
            return f

        def g(wl):
            ww = np.zeros((4, 6)) # change to 3,6 without sun # To 4,6 with sun
            for i in range(particles): #add 1,particles for no sun # only particles with sun
                wa = wl[i][0:3]
                va = wl[i][3:6]
                flist = [[],[],[]]
                for j in range(particles):
                    if j!= i:
                        F = np.nan_to_num(Force(wa, wl[j][0:3], Mass[j]))
                        for x in range(3):
                            flist[x].append(F[x])
                fan = [np.sum(flist[x]) for x in range(3)]
                out = [va[0],va[1],va[2]]
                out[len(fan):] = fan # add the velocities to matrix
                ww[i] = out #add -1 for no sun

            return ww

        def rungekutta(wl,dt, low, high):
            t = np.arange(low, high, dt)
            W_a = np.zeros((len(t), len(wl), 6))
            W_a[0] = wl
            distlist = np.zeros((len(t)-1, particles-1))
            for i in range(1, len(t)):
                fa = g(wl)
                wb = wl + fa*dt/2
                fb = g(wb)
                wc = wl + (dt/2)*fb
                fc = g(wc)
                wd = wl + (dt)*fc
                fd = g(wd)
                www = wl + (dt/6)*fa + (dt/3)*fb + (dt/3)*fc + (dt/6)*fd
                W_a[i] = www - www[0]
                wl = www

            jupiter = [[px,py] for px,py in zip(W_a[:,1][:,0],W_a[:,1][:,1])] 
            troj1 = [[px,py] for px,py in zip(W_a[:,2][:,0],W_a[:,2][:,1])]
            troj2 = [[px,py] for px,py in zip(W_a[:,3][:,0],W_a[:,3][:,1])]

            return jupiter, troj1, troj2
        jupiter, troj1, troj2 = rungekutta(W,0.01,0,200)
        points = self.get_points_from_coords(jupiter)
        points2  = self.get_points_from_coords(troj1)
        points3 = self.get_points_from_coords(troj2)
        
        # Define graph and animation properties from class
        graph = SmoothGraphFromSetPoints(points,color=GREEN)
        graph2 = SmoothGraphFromSetPoints(points2, color=RED)
        graph3 = SmoothGraphFromSetPoints(points3, color = YELLOW)

        p = Dot(color = RED)
        p.scale(2)
        p2 = Dot(color = BLUE)
        p3 = Dot(color = YELLOW)
        p4 = Dot(color = ORANGE)
        p4.scale(5)

        t = TextMobject('yr')

        tracker = ValueTracker(0)
        numbers = DecimalNumber(0)
        numbers.add_updater(lambda m: m.set_value(tracker.get_value()))
        numbers.to_edge(LEFT+ 2*UP)
        t.next_to(numbers, 3*RIGHT)

        Heading = TextMobject('Jupiter and Trojans')
        Heading.to_edge(UP + LEFT)

        sun = TextMobject('Sun')
        sun.scale(0.6)
        sun.next_to(p4, buff = TOP)
        p4.move_to([0,0,0])
        jup = TextMobject('Jupiter')
        jup.scale(0.6)
        jup.next_to(p, buff = LEFT)
        troj = TextMobject('Trojan 1')
        troj.scale(0.6)
        troj.next_to(p2, buff = LEFT)
        troj2 = TextMobject('Trojan 2')
        troj2.scale(0.6)
        troj2.next_to(p3, buff = LEFT)
        self.add(numbers)
        self.play(Write(p4), Write(sun), Write(Heading), Write(t))
        self.play(MoveAlongPath(p,graph, rate_func = linear), MoveAlongPath(p2, graph2, rate_func = linear),MoveAlongPath(p3, graph3, rate_func = linear), MoveAlongPath(jup,p),MoveAlongPath(troj,p2),MoveAlongPath(troj2,p3), tracker.set_value, 200, rate_func = linear, run_time=200)
        self.wait()


class solarsystem(GraphFromData):

    CONFIG = {
        "y_max": 3,
        "x_max": 3,
        "y_tick_frequency" : 1,
        "x_tick_frequency" : 1,
        "y_min": -3,
        "x_min": -3,
        "graph_origin": 0,
        "y_labeled_nums": list(np.arange(0,34,1)),
        "x_labeled_nums": list(np.arange(0,4,1)),
    }
    def construct(self):
        #grid = ScreenGrid()
        self.setup_axes(animate=True)

        from astropy . coordinates import solar_system_ephemeris , EarthLocation
        from astropy . coordinates import get_body_barycentric , get_body , get_moon ,get_body_barycentric_posvel
        from astropy . time import Time
        import numpy as np

        planets = ['earth',
                'mercury',
        'venus',
        'mars',
        'jupiter',
        'saturn',
        'uranus',
        'neptune']

        pl = []
        t = Time("2021-02-01 15:48")
        for i in planets:
            pos = get_body_barycentric_posvel(i,time = t, ephemeris='builtin')  
            pl.append(pos)
        Masslist = []
        xvals = []
        yvals = []
        zvals = []
        vxvals = []
        vyvals = []
        vzvals = []
        for j in range(len(planets)):
            xvals.append(pl[j][0].x.value)
            yvals.append(pl[j][0].y.value)
            zvals.append(pl[j][0].z.value)
            vxvals.append(pl[j][1].x.value*365)
            vyvals.append(pl[j][1].y.value*365)
            vzvals.append(pl[j][1].z.value*365)

        w_sun = [0, 0, 0, 0, 0, 0]
        w_earth = [xvals[0],yvals[0],zvals[0],vxvals[0],vyvals[0],vzvals[0]]
        w_merc = [xvals[1],yvals[1],zvals[1],vxvals[1],vyvals[1],vzvals[1]]
        w_ven = [xvals[2],yvals[2],zvals[2],vxvals[2],vyvals[2],vzvals[2]]
        w_mars = [xvals[3],yvals[3],zvals[3],vxvals[3],vyvals[3],vzvals[3]]
        w_jup = [xvals[4],yvals[4],zvals[4],vxvals[4],vyvals[4],vzvals[4]]
        w_sat = [xvals[5],yvals[5],zvals[5],vxvals[5],vyvals[5],vzvals[5]]
        w_ura = [xvals[6],yvals[6],zvals[6],vxvals[6],vyvals[6],vzvals[6]]
        w_nep = [xvals[7],yvals[7],zvals[7],vxvals[7],vyvals[7],vzvals[7]]
        #w_x = [4.6, -39.9, 0, 0.06324, 1.338, 0]
        W = np.array([w_sun, w_earth, w_merc, w_ven, w_mars, w_jup, w_sat, w_ura, w_nep])
        msun, mearth, mmerc, mven, mmars, mjup, msat, mura, mnep = [
            1, #sun
            1/333030, # Earth
            1.651e-7, # Mercury
            0.000002447, # Venus
            0.0000003213, # Mars
            0.000954588, # Jupyter
            0.0002857, # Saturn
            0.00004365, # Uranus
            0.00005149 # Neptune,
        ]

        Mass = [msun, mearth, mmerc, mven, mmars, mjup, msat, mura, mnep]

        particles = len(W) #Can remove this by looking at length of W matrix

        G = 4*np.pi**2
        def Force(xi, xj, mass_j):
            rad = [(xi[i] - xj[i]) for i in range(3)]
            absrad = ((rad[0])**2 + (rad[1])**2 + (rad[2])**2)**(0.5)
            f = [-G * ((mass_j)*(rad[i]))/((absrad)**3) for i in range(3)]
            return f


        def g(wl):
            ww = np.zeros((len(W), 6)) # change to 3,6 without sun # To 4,6 with sun
            for i in range(particles): #add 1,particles for no sun # only particles with sun
                wa = wl[i][0:3]
                va = wl[i][3:6]
                flist = [[],[],[]]
                for j in range(particles):
                    if j!= i:
                        F = np.nan_to_num(Force(wa, wl[j][0:3], Mass[j]))
                        for x in range(3):
                            flist[x].append(F[x])
                fan = [np.sum(flist[x]) for x in range(3)]
                out = [va[0],va[1],va[2]]
                out[len(fan):] = fan # add the velocities to matrix
                ww[i] = out #add -1 for no sun

            return ww



        def rungekutta(wl,dt, solar = True, oscillation = False):
            t0 = 0
            tn = 120
            t = np.arange(t0, tn+dt, dt)
            W_a = np.zeros((len(t), W.shape[0], W.shape[1]))
            W_a[0] = wl
            distlist = np.zeros((len(t)-1, particles-1))
            zlist = np.zeros((len(t)-1, particles-1))
            for i in range(1, len(t)):
                fa = g(wl)
                wb = wl + fa*dt/2
                fb = g(wb)
                wc = wl + (dt/2)*fb
                fc = g(wc)
                wd = wl + (dt)*fc
                fd = g(wd)
                www = wl + (dt/6)*fa + (dt/3)*fb + (dt/3)*fc + (dt/6)*fd
                W_a[i] = www - www[0]
                wl = www

            earth = [[px,py] for px,py in zip(W_a[:,1][:,0],W_a[:,1][:,1])] 
            merc = [[px,py] for px,py in zip(W_a[:,2][:,0],W_a[:,2][:,1])]
            ven = [[px,py] for px,py in zip(W_a[:,3][:,0],W_a[:,3][:,1])]
            Mars = [[px,py] for px,py in zip(W_a[:,4][:,0],W_a[:,4][:,1])]
            Jupiter = [[px,py] for px,py in zip(W_a[:,5][:,0],W_a[:,5][:,1])]
            Saturn = [[px,py] for px,py in zip(W_a[:,6][:,0],W_a[:,6][:,1])]
            Uranus = [[px,py] for px,py in zip(W_a[:,7][:,0],W_a[:,7][:,1])]
            Neptune = [[px,py] for px,py in zip(W_a[:,8][:,0],W_a[:,8][:,1])]

            return earth, merc, ven, Mars, Jupiter, Saturn, Uranus, Neptune
        earth, merc, ven, Mars, Jupiter, Saturn, Uranus, Neptune = rungekutta(W,0.01)

        points = self.get_points_from_coords(earth)
        points2  = self.get_points_from_coords(merc)
        points3 = self.get_points_from_coords(ven)
        points4 = self.get_points_from_coords(Mars)
        points5 = self.get_points_from_coords(Jupiter)
        points6 = self.get_points_from_coords(Saturn)
        points7 = self.get_points_from_coords(Uranus)
        points8 = self.get_points_from_coords(Neptune)
        
        # Define graph and animation properties from class
        graph = SmoothGraphFromSetPoints(points,color=GREEN) #earth
        graph2 = SmoothGraphFromSetPoints(points2, color=RED) # Mercury
        graph3 = SmoothGraphFromSetPoints(points3, color = BLUE) # Venus
        graph4 = SmoothGraphFromSetPoints(points4) # Mars
        graph5 = SmoothGraphFromSetPoints(points5) # Jupiter
        graph6 = SmoothGraphFromSetPoints(points6) # Saturn
        graph7 = SmoothGraphFromSetPoints(points7) # Uranus
        graph8 = SmoothGraphFromSetPoints(points8) # Neptune

        p = Dot(color = GREEN) #earth
        p2 = Dot(color = RED) # Mercury
        p3 = Dot(color = BLUE) # Venus
        p4 = Dot(color = ORANGE) # Sun
        p4.scale(2)
        p5 = Dot(color = YELLOW) # Mars
        p6 = Dot(color = GREY_BROWN) # Jupiter
        p7 = Dot()
        p8 = Dot(color = GOLD)
        p9 = Dot(color = TEAL)

        t = TextMobject('yr')

        tracker = ValueTracker(0)
        numbers = DecimalNumber(0)
        numbers.add_updater(lambda m: m.set_value(tracker.get_value()))
        numbers.to_edge(LEFT+ 2*UP)
        t.next_to(numbers, 3*RIGHT)

        Heading = TextMobject('Inner solar System')
        Heading.to_edge(UP + LEFT)

        sun = TextMobject('Sun')
        sun.scale(0.6)
        sun.next_to(p4, buff = TOP)
        p4.move_to([0,0,0])

        ear = TextMobject('earth')
        ear.scale(0.6)
        ear.next_to(p, buff = LEFT)

        mer = TextMobject('Mercury')
        mer.scale(0.6)
        mer.next_to(p2, buff = LEFT)

        venus = TextMobject('Venus')
        venus.scale(0.6)
        venus.next_to(p3, buff = LEFT)

        Mar = TextMobject('Mars')
        Mar.scale(0.6)
        Mar.next_to(p5, buff = LEFT)

        Jup = TextMobject('Jupiter')
        Jup.scale(0.6)
        Jup.next_to(p6, buff = LEFT)

        Sat = TextMobject('Saturn')
        Sat.scale(0.6)
        Sat.next_to(p6, buff = LEFT)

        Ura = TextMobject('Uranus')
        Ura.scale(0.6)
        Ura.next_to(p6, buff = LEFT)

        Nep = TextMobject('Neptune')
        Nep.scale(0.6)
        Nep.next_to(p6, buff = LEFT)

        self.add(numbers)
        self.play(Write(p4), Write(sun), Write(Heading), Write(t))
        self.play(MoveAlongPath(p,graph, rate_func = linear), MoveAlongPath(ear,p) ,MoveAlongPath(p2, graph2, rate_func = linear), MoveAlongPath(mer,p2), MoveAlongPath(p3, graph3, rate_func = linear), MoveAlongPath(venus,p3), MoveAlongPath(p5, graph4, rate_func = linear) , MoveAlongPath(Mar,p5), MoveAlongPath(p5, graph4, rate_func = linear),tracker.set_value, 120, rate_func = linear,run_time=200)#, MoveAlongPath(p6, graph5, rate_func = linear), MoveAlongPath(p7, graph6, rate_func = linear), MoveAlongPath(p8, graph7, rate_func = linear), MoveAlongPath(p9, graph8, rate_func = linear),MoveAlongPath(Jup, p6),MoveAlongPath(Sat, p7), MoveAlongPath(Ura, p8), MoveAlongPath(Nep, p9),tracker.set_value, 200, rate_func = linear,run_time=200) #, MoveAlongPath(ear,p),MoveAlongPath(mer,p2),MoveAlongPath(venus,p3),MoveAlongPath(Mar,p5),MoveAlongPath(Jup,p6),run_time = 10)
        self.wait()
        