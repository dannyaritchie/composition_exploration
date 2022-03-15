import matplotlib.pyplot as plt
import ternary
import numpy as np

SQRT3 = np.sqrt(3)
SQRT3OVER2 = SQRT3 / 2.

class Plotter:

    def __init__(self,which):
        self.which=which
    
    def add_points(self,points_in_standard_basis):
        if self.which=='jon':
            self.points=np.empty((0,3))
            if self.dim != 5:
                print('Error, jon special requires 5 elements')
            for j in points_in_standard_basis:
                x = np.empty((3))
                x[0]=0.5*(2*j[3]+j[4])/(j[2]+j[3]+j[4])
                x[1]=SQRT3OVER2*j[4]/(j[2]+j[3]+j[4])
                x[2]=j[0]/(j[1]+j[0])
                self.points=np.append(self.points,[x],axis=0)
        if self.which=='berny':
            self.points=points_in_standard_basis[:,:3]

    def ternary_project(self,p):
        #projects 3d p to 2d simplex
        x=p[0]+p[1]/2
        y=SQRT3OVER2*p[1]
        return np.array([x,y])

    def berny_testing(self,data,points,end_points,point_labels=None):
        fig, axs = plt.subplots(1,2)
        taxa=self.create_fig(ax=axs[0])
        self.plot_heatmap_ternary(data, tax=taxa)
        taxb=self.create_fig(ax=axs[1])
        self.points=points
        self.point_labels=point_labels
        self.plot_points(tax=taxb,use_labels=True,s=10)
        self.plot_lines(points,end_points,taxb,linewidth=0.9)
        self.show_legend(taxb)
        plt.show()

    def create_fig(self,ax=None,scale=100):
        if self.which == 'jon':
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(projection='3d')
            self.set_axes(color='black')
            self.set_horizontal_ticks()
            self.set_vertical_ticks()
            self.set_labels()
        if self.which=='berny':
            if ax is None:
                fig=plt.figure()
                ax=fig.subplots(1,1)
            self.set_axes(ax=ax,color='black')
            self.set_horizontal_ticks(ax=ax)
            plt.show()
        if self.which=='bernyterny':
            if ax is None:
                figure,tax=ternary.figure(scale=scale)
            else:
                figure, tax=ternary.figure(ax=ax,scale=scale)
                tax.clear_matplotlib_ticks()
            tax.boundary(linewidth=1)
            tax.gridlines(color="black", multiple=10)
            tax.ticks(axis='lbr', linewidth=1, multiple = 10, fontsize=5)
            if ax is None:
                self.tax=tax
                self.figure=figure
            else:
                return tax

    def show_legend(self,tax):
            ax=tax.get_axes()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    def plot_heatmap_ternary(self,data,tax=None,scale=100):
        if tax is None:
            self.tax.heatmap(data,scale)
        else:
            tax.heatmap(data,scale)
            return tax

    def set_axes(self,ax=None,**kwargs):
        plt.axis('off')
        if self.which=='jon':
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
        if self.which == 'berny':
            top=[1,0,0]
            left=[0,1,0]
            right=[0,0,1]
            self.line_ternary(top,left,ax=ax,**kwargs)
            self.line_ternary(right,left,ax=ax,**kwargs)
            self.line_ternary(right,top,ax=ax,**kwargs)

    def set_labels(self,offset=0.2):
        label='Cl'
        position=[-offset/2,1+offset,0]
        position=self.ternary_project(position)
        self.ax.text(position[0],position[1],0,label)
        label='Se'
        position=[-0.5*offset,-0.5*offset,0]
        position=self.ternary_project(position)
        self.ax.text(position[0],position[1],0,label)
        label='I'
        position=[1,offset/2,0]
        position=self.ternary_project(position)
        self.ax.text(position[0],position[1],0,label)
        label='Cs/(Cs+Bi)'
        position=[-1.5*offset,-1.5*offset,0]
        position=self.ternary_project(position)
        self.ax.text(position[0],position[1],0.5,label)

    def set_horizontal_ticks(self, scale=1, multiple= 0.1,
                  tick_formats='%d', offset=0.02,
                  axes_colors='black',
                  fontsize=5,z=0,ax=None
                 ):
        if self.which=='berny':
            locations = np.arange(0, scale + multiple, multiple)
            ticks = locations
            offset*=scale
            for index,i in enumerate(locations):
                loc1 = (scale-i,i,0)
                text_location = (scale - i, i + 4 * offset, 0)
                loc2 = (scale-i,i+offset,0)
                self.line_ternary(loc1,loc2,ax,color=axes_colors)
                x,y=self.ternary_project(text_location)
                tick = ticks[-(index+1)]*10
                ax.text(x, y, '%d' % tick, horizontalalignment="center",
                        color=axes_colors, fontsize=fontsize)
            for index,i in enumerate(locations):
                loc1 = (0,i,0)
                text_location = (-2*offset, i-0.5*offset, 0)
                loc2 = (-offset,i,0)
                self.line_ternary(loc1,loc2,ax,color=axes_colors)
                x,y=self.ternary_project(text_location)
                tick = ticks[index]*10
                ax.text(x, y, '%d' % tick, horizontalalignment="center",
                        color=axes_colors, fontsize=fontsize)
            for index,i in enumerate(locations):
                loc1 = (i,0,0)
                text_location = (i+3*offset, -3.5*offset, 0)
                loc2 = (i+offset,-offset,0)
                self.line_ternary(loc1,loc2,ax,color=axes_colors)
                x,y=self.ternary_project(text_location)
                tick = ticks[-(index+1)]*10
                ax.text(x, y, '%d' % tick, horizontalalignment="center",
                        color=axes_colors, fontsize=fontsize)
        if self.which=='jon':
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

    def set_vertical_ticks(self,scale=1,multiple=0.1,
                           tick_formats='%d',offset=0.02,axes_colors='black',
                           fontsize=5
                          ):
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
            self.ax.text(x, y, i-2*offset, '%d' % tick, horizontalalignment="center",
                    color=axes_colors, fontsize=fontsize)

    def line(self,p1,p2,z1,z2,ax=None,**kwargs):
        if self.which=='jon':
            pp1=self.ternary_project(p1)
            pp2=self.ternary_project(p2)
            self.ax.plot([pp1[0],pp2[0]],[pp1[1],pp2[1]],[z1,z2],**kwargs)

    def line_ternary(self,p1,p2,ax,**kwargs):
        if self.which =='berny':
            pp1=self.ternary_project(p1)
            pp2=self.ternary_project(p2)
            ax.plot([pp1[0],pp2[0]],[pp1[1],pp2[1]],**kwargs)
            
        

    def plot_points(self,show=True,use_labels=False,tax=None,**kwargs):
        if self.which=='jon':
            for i in self.points:
                self.ax.scatter(i[0],i[1],i[2],**kwargs)
            if show:
                if self.show_labels:
                    self.ax.legend(prop={'size':5})
                plt.show()
        if self.which=='bernyterny':
            if not use_labels:
                if tax is not None:
                    tax.scatter(self.points)
            else:
                if tax is not None:
                    for key in self.point_labels:
                        start=self.point_labels[key][0]
                        stop=self.point_labels[key][1]
                        tax.scatter(self.points[start:stop],label=key,**kwargs)
            return tax

    def plot_lines(self,starts,finishes,tax,**kwargs):
        for start,finish in zip(starts,finishes):
            tax.line(start,finish,**kwargs)
        return tax


    def show_labels(self):
       self.show_labels=True

    def set_projection(self,elev,azim):
        self.ax.view_init(elev=elev,azim=azim)
        


