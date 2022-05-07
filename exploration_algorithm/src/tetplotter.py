import matplotlib.pyplot as plt
import ternary
import numpy as np

SQRT3 = np.sqrt(3)
SQRT3OVER2 = SQRT3 / 2.

class tetPlotter:

    def __init__(self):
        self.create_fig()

    def add_points(self,points_in_standard_basis):
        self.points=np.empty((0,3))
        for j in points_in_standard_basis:
            x = np.empty((3))
            x[0]=0.5*(2*j[3]+j[4])/(j[2]+j[3]+j[4])
            x[1]=SQRT3OVER2*j[4]/(j[2]+j[3]+j[4])
            x[2]=j[0]/(j[1]+j[0])
            self.points=np.append(self.points,[x],axis=0)

    def set_directory(self,directory):
        self.directory=directory

    def create_fig(self,ax=None,scale=100):
        self.fig=plt.figure()
        self.ax=self.fig.add_subplot(projection='3d')
        self.set_axes(color='black')
        self.set_horizontal_ticks()
        self.set_vertical_ticks()
        self.set_labels()

    def set_axes(self,ax=None,**kwargs):
        plt.axis('off')
        bl1 = [0,0,1]
        bl2 = [1,0,0]
        rl1 = [0,1,0]
        rl2 = [1,0,0]
        ll1 = [0,1,0]
        ll2 = [0,0,1]
        z1=0
        z2=1
        self.line(rl1,rl1,0,1,**kwargs)
        self.line(rl2,rl2,0,1,**kwargs)
        self.line(bl1,bl1,0,1,**kwargs)
        for z in [z1,z2]:
            self.line(bl1,bl2,z,z,**kwargs)
            self.line(rl1,rl2,z,z,**kwargs)
            self.line(ll1,ll2,z,z,**kwargs)

    def set_labels(self,offset=0.2):
        label='I'
        position=[-offset/2,1+offset,0]
        position=self.ternary_project(position)
        self.ax.text(position[0],position[1],0,label)
        label='Se'
        position=[-0.5*offset,-0.5*offset,0]
        position=self.ternary_project(position)
        self.ax.text(position[0],position[1],0,label)
        label='Cl'
        position=[1,offset/2,0]
        position=self.ternary_project(position)
        self.ax.text(position[0],position[1],0,label)
        label='Cs/(Cs+Bi)'
        position=[-1.5*offset,-1.5*offset,0]
        position=self.ternary_project(position)
        self.ax.text(position[0],position[1],0.5,label)

    def set_horizontal_ticks(
            self,scale=1,multiple= 0.1,tick_formats='%d',offset=0.02,
            axes_colors='black',fontsize=5,z=0,ax=None):
        locations = np.arange(0, scale + multiple, multiple)
        ticks = locations
        offset*=scale
        for index,i in enumerate(locations):
            loc1 = (scale-i,i,0)
            text_location = (scale - i, i + 4 * offset, 0)
            loc2 = (scale-i,i+offset,0)
            self.line(loc1,loc2,z,z,color=axes_colors)
            x,y=self.ternary_project(text_location)
            tick = ticks[-(index+1)]*10
            self.ax.text(x, y, z, '%d' % tick, horizontalalignment="center",
                    color=axes_colors, fontsize=fontsize)
        for index,i in enumerate(locations):
            loc1 = (0,i,0)
            text_location = (-2*offset, i-0.5*offset, 0)
            loc2 = (-offset,i,0)
            self.line(loc1,loc2,z,z,color=axes_colors)
            x,y=self.ternary_project(text_location)
            tick = ticks[index]*10
            self.ax.text(x, y, z, '%d' % tick, horizontalalignment="center",
                    color=axes_colors, fontsize=fontsize)
        for index,i in enumerate(locations):
            loc1 = (i,0,0)
            text_location = (i+3*offset, -3.5*offset, 0)
            loc2 = (i+offset,-offset,0)
            self.line(loc1,loc2,z,z,color=axes_colors)
            x,y=self.ternary_project(text_location)
            tick = ticks[-(index+1)]*10
            self.ax.text(x, y, z, '%d' % tick, horizontalalignment="center",
                    color=axes_colors, fontsize=fontsize)

    def set_vertical_ticks(
            self,scale=1,multiple=0.1,tick_formats='%d',offset=0.02,
            axes_colors='black',fontsize=5):
        locations = np.arange(0, scale + multiple, multiple)
        ticks = locations
        offset*=scale
        for index,i in enumerate(locations):
            loc1 = (0-offset,0,0)
            loc2 = (0,0,0)
            self.line(loc1,loc2,i,i,color=axes_colors)
            text_location = (0-3*offset, 0, 0)
            x,y=self.ternary_project(text_location)
            tick = ticks[index]*10
            self.ax.text(
                x, y, i-2*offset, '%d' % tick, horizontalalignment="center",
                color=axes_colors, fontsize=fontsize)

    def plot_points(
            self,use_labels=False,tax=None,labels=None,**kwargs):
        self.ax.scatter(
            self.points[:,0],self.points[:,1],self.points[:,2],**kwargs)

    def line(self,p1,p2,z1,z2,ax=None,**kwargs):
        pp1=self.ternary_project(p1)
        pp2=self.ternary_project(p2)
        self.ax.plot([pp1[0],pp2[0]],[pp1[1],pp2[1]],[z1,z2],**kwargs)

    def ternary_project(self,p):
        #projects 3d p to 2d simplex
        x=p[0]+p[1]/2
        y=SQRT3OVER2*p[1]
        return np.array([x,y])

    def plot_plane(self,points):

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 10 * np.outer(np.cos(u), np.sin(v))
        y = 10 * np.outer(np.sin(u), np.sin(v))
        z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
        print('hhhhhhhhhh')
        print(x.shape)
        print(z)

        x=0.5*(2*points[:,3]+points[:,4])/(points[:,2]+points[:,3]+points[:,4])
        X=np.broadcast_to(x,(len(x),len(x)))
        y=SQRT3OVER2*points[:,4]/(points[:,2]+points[:,3]+points[:,4])
        Y=np.broadcast_to(x,(len(x),len(x)))
        z=points[:,0]/(points[:,1]+points[:,0])
        Z=np.broadcast_to(x,(len(x),len(x)))
        surf = self.ax.plot_trisurf(x, y, z,linewidth=0)



    def set_projection(self,elev,azim):
        self.ax.view_init(elev=elev,azim=azim)
    def legend(self):
        self.ax.legend()

