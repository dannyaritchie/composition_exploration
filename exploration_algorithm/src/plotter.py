import matplotlib.pyplot as plt
import ternary
import numpy as np

SQRT3 = np.sqrt(3)
SQRT3OVER2 = SQRT3 / 2.

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

            
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
        self.plot_heatmap_ternary(data,taxa)
        taxb=self.create_fig(ax=axs[1])
        self.points=points
        self.point_labels=point_labels
        self.plot_points(tax=taxb,use_labels=True,s=10)
        self.plot_lines(points,end_points,taxb,linewidth=0.9)
        self.show_legend(taxb)
        plt.show()

    def small_balls_heat_fig(self,data,scale=100):
        figure,tax=ternary.figure(scale=scale)
        tax.boundary(linewidth=1)
        tax.gridlines(color="black", multiple=10)
        tax.ticks(axis='lbr', linewidth=1, multiple = 10, fontsize=5)
        self.plot_heatmap_ternary(data,tax)
        plt.show()

    def tax_setup(self,tax,method):
        fontsize=10
        tax.top_corner_label('ZnS',fontsize=fontsize,offset=0.15)
        tax.right_corner_label('Li$_2$S',fontsize=fontsize,offset=0.05)
        tax.left_corner_label('SiS',fontsize=fontsize,offset=0.05)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        tax.boundary(linewidth=0.5)

    def set_aspect(self,fig,tax,kind="",labels=True):
        ax=tax.get_axes()
        box = ax.get_position()
        if kind!='heat':
            ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])
        if labels:
            ax.legend(loc='upper left', bbox_to_anchor=(0.68, 1.05))
        ax.set_aspect(1)
        tax._redraw_labels()
        fig.set_size_inches(5,5)

    def set_scatter_kwargs(self):
        self.scatter_ka={'marker':'x',
                         'linewidth':0.5,
                         's':20,
                         'zorder':8,
                     }

    def set_heat_cbar_kwargs(self):
        self.cb_kwargs = {"shrink" : 0.75,
                     #"orientation" : "horizontal",
                     "fraction" : 0.1,
                     "pad" : 0.02,
                     "aspect" : 25,
                     "anchor":(0,0.7)
                    }

    def set_directory(self,directory):
        self.directory=directory

    def mean_line(self,points,end_points,mean):
        fig, tax= ternary.figure(scale=100)
        self.tax_setup(tax,'line')
        tax.scatter(
            points,label='Initial',color='Blue',**self.scatter_ka)
        tax.scatter(
            [mean],label='K',color='Green',**self.scatter_ka)
        tax.line(mean,points[0],linestyle='--',linewidth=0.5,color='Green')
        tax.line(points[0],end_points[0],linewidth=0.5,color='Blue')
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + "mean_line.png")
        #plt.show()

    def mean_small(self,mean,small_means,labels):
        fig, tax= ternary.figure(scale=100)
        self.tax_setup(tax,'line')
        colors=['orange','turquoise','purple']
        tax.scatter(
            [mean],label='K',color='Green',**self.scatter_ka)
        for point,label,c in zip(small_means,labels,colors):
            tax.scatter(
                [point],label=label,color=c,**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + "mean_small.png")
        #plt.show()

    def mean_all(self,initial,il,small_means,labels,goal,gl):
        fig, tax=ternary.figure(scale=100)
        self.tax_setup(tax,'line')
        colors=['green','red','orange']
        for point,label,c in zip(small_means,labels,colors):
            tax.scatter(
                [point],label=label,color=c,**self.scatter_ka)
        tax.scatter([goal],label=gl,color='purple',**self.scatter_ka)
        tax.scatter([initial],label='Initial sample',color='blue',**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + "mean_all.png")
        plt.show()

    def merged_ball(self,data,mean):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter(
            [mean],label='K',color='Green',**self.scatter_ka)
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat')
        plt.savefig(self.directory + "mergedball.png")
        #plt.show()

    def p_mean_initial(self,data,mean,points):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter(
            [mean],label='K',color='Green',**self.scatter_ka)
        tax.scatter(
            points,label='Initial',color='Blue',**self.scatter_ka)
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat')
        plt.savefig(self.directory + "p mean initial.png")
        #plt.show()

    def linebatch_initial(self,points,labels,colors):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        for point,label,color in zip(points,labels,colors):
            tax.scatter(
                [point],label=label,color=color,**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + "linebatch_initial.png")
        #plt.show()

    def linebatch_initial_chosen(self,points,labels,colors,chosen_point):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter(
            [chosen_point],marker='o',label='Closest',color='red',zorder=0,
            s=30)
        for point,label,color in zip(points,labels,colors):
            tax.scatter(
                [point],color=color,**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + "linebatch_initial_chosen.png")
        #plt.show()

    def first_chosen(self,ps,es,ls,cs):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,method='')
        for (p,e,l,c) in zip(ps,es,ls,cs):
            tax.scatter([p],label=l,color=c,**self.scatter_ka)
            tax.line(p,e,linewidth=0.5,color=c,zorder=7)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory+'first_chosen.png',dpi=200)
        #plt.show()

    def p_second_max(self,data,ps,es,ls,cs):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat')
        plt.savefig(self.directory + "p_second_max.png")
        #plt.show()

    def p_second(self,data):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat',labels=False)
        plt.savefig(self.directory + 'p_second.png',dpi=200)
        #plt.show()

    def p_second_maxi_test(self,data,ps,es,ls,cs,mean,goal):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        tax.scatter([mean],label='mean',color='lime',**self.scatter_ka)
        tax.scatter([goal],label='goal',color='lightblue',**self.scatter_ka)
        for (p,e,l,c) in zip(ps,es,ls,cs):
            tax.scatter([p],label=l,color=c,**self.scatter_ka)
            tax.line(p,e,linewidth=0.5,color=c,zorder=7)
        self.set_aspect(fig,tax,kind='heat')
        plt.show()

    def second_batch(self,points):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter(
            points,label='Batch 2',color='Lime',**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + 'second_batch.png',dpi=200)
        #plt.show()

    def second_chosen(self,ps,es,ls,cs):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,method='')
        for (p,e,l,c) in zip(ps,es,ls,cs):
            tax.scatter([p],label=l,color=c,**self.scatter_ka)
            tax.line(p,e,linewidth=0.5,color=c,zorder=7)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory+'second_chosen.png',dpi=200)
        #plt.show()

    def p_third(self,data):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat',labels=False)
        plt.savefig(self.directory+'p_third.png',dpi=200)
        #plt.show()

    def third_batch(self,points):
        fig,tax=ternary.figure(scale=100)
        ka=self.scatter_ka
        self.tax_setup(tax,'heat')
        tax.scatter(
            points,label='Batch 3',color='Turquoise',**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + 'third_batch.png',dpi=200)
        #plt.show()

    def third_closest(self,chosen):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter(
            [chosen],label='3rd Closest',color='Turquoise',**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + 'third_closest.png',dpi=200)
        #plt.show()

    def final(self,closest,goal):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter([goal],label='U',color='Purple',**self.scatter_ka)
        tax.scatter([closest],label='3rd Closest',color='Turquoise',**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + 'final.png',dpi=200)
        #plt.show()

    def final_testa(self,data,goal):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter([goal],label='U',color='Purple',**self.scatter_ka)
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat')
        plt.savefig(self.directory + 'finala.png',dpi=200)
        #plt.show()

    def final_testb(self,points,labels,colors,goal):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        for p,l,c in zip(points,labels,colors):
            tax.scatter(
                [p],label=l,color=c,**self.scatter_ka)
        tax.scatter([goal],label='U',color='Purple',**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + 'finalb.png',dpi=200)
        #plt.show()

    def process_a(self,goal,num_samples):
        self.cmap=get_cmap(num_samples+3)
        self.goal=goal

    def process_batch(self,goal,num_batches):
        self.cmap=get_cmap(num_batches+3)
        self.goal=goal

    def explore_batch(self,goal,data_initial):
        self.cmap=get_cmap(5)
        self.goal=goal
        self.pre_data=data_initial

    def explore_post_data(self,exp_points,exp_targets,points,end_points,data):
        self.exploration_points=exp_points
        self.exploration_targets=exp_targets
        self.points=points
        self.end_points=end_points
        self.post_data=data

    def explore_plot(self,filename):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'lines')
        tax.scatter(
            self.points[0:1],marker='x',linewidth=1,color=self.cmap(0),
            label='Initial'
        )
        tax.scatter(
            [self.goal],marker='x',linewidth=1,color=self.cmap(1),
            label='Goal'
        )
        tax.line(self.points[0],self.end_points[0],linewidth=1,color=self.cmap(0))
        ax=tax.get_axes()
        ax.legend(loc='upper right')
        plt.savefig(filename + 'a.png',bbox_inches='tight')
        plt.clf()
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.heatmap(self.pre_data,100)
        tax.scatter(
            self.exploration_points,marker='x',linewidth=1,color='red',
            label='ppoints',zorder=8,s=2)
        tax.scatter(
            self.exploration_targets,marker='x',linewidth=1,color='orange',
            label='ptargets',zorder=8,s=2)
        plt.savefig(filename + 'b.png',bbox_inches='tight')
        plt.clf()
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'lines')
        tax.scatter(
            self.points[0:1],marker='x',linewidth=1,color=self.cmap(0),
            label='Initial'
        )
        tax.scatter(
            [self.goal],marker='x',linewidth=1,color=self.cmap(1),
            label='Goal'
        )
        tax.scatter(
            self.points[1:],marker='x',linewidth=1,color=self.cmap(2),
            label='Chosen points'
        )
        tax.line(self.points[0],self.end_points[0],linewidth=1,color=self.cmap(0))
        for i in range(1,len(self.points)):
            tax.line(self.points[i],self.end_points[i],linewidth=1,
                     color=self.cmap(2))
        ax=tax.get_axes()
        ax.legend(loc='upper right')
        plt.savefig(filename + 'c.png',bbox_inches='tight')
        plt.clf()
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.heatmap(self.post_data,100)
        tax.scatter(
            [self.goal],marker='x',linewidth=1,color='red',
            label='ppoints',zorder=8,s=2)
        plt.savefig(filename + 'd.png',bbox_inches='tight')
        

    def berny_heat_points(self,filename,data,points,labels,end_points):
        fig,axs=plt.subplots(1,2,figsize=(13.33,7.5),dpi=400)
        fontsize=10
        fig,tax=ternary.figure(scale=100,ax=axs[0])
        #tax.boundary(linewidth=2.0)
        tax.gridlines(color="black", multiple=10)
        tax.left_axis_label('SiS',fontsize=fontsize)
        tax.right_axis_label('ZnS',fontsize=fontsize)
        tax.bottom_axis_label('Li2S',fontsize=fontsize,offset=0)
        tax.ticks(axis='lbr', linewidth=1, multiple = 10, fontsize=5)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        self.plot_heatmap_ternary(data,tax)
        figb,taxb=ternary.figure(scale=100,ax=axs[1])
        self.plot_points_berny(taxb,points,labels)
        self.plot_goal_berny(taxb)
        self.plot_lines(points,end_points,taxb,labels,linewidth=0.9)
        taxb.gridlines(color="black", multiple=10)
        taxb.left_axis_label('SiS',fontsize=fontsize)
        taxb.right_axis_label('ZnS',fontsize=fontsize)
        taxb.bottom_axis_label('Li2S',fontsize=fontsize,offset=0)
        taxb.ticks(axis='lbr', linewidth=1, multiple = 10, fontsize=5)
        taxb.clear_matplotlib_ticks()
        taxb.get_axes().axis('off')
        axb=taxb.get_axes()
        box = axb.get_position()
        axb.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        axb.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
        plt.savefig(filename,bbox_inches='tight')

    def merged_balls(
        self,data,merged_mean,small_means,point_labels,mean_label,scale=100):
        figure,tax=ternary.figure(scale=scale)
        fontsize=10
        #tax.boundary(linewidth=2.0)
        tax.gridlines(color="black", multiple=10)
        tax.set_title('Average compostion')
        tax.left_axis_label('SiS',fontsize=fontsize)
        tax.right_axis_label('ZnS',fontsize=fontsize)
        tax.bottom_axis_label('Li2S',fontsize=fontsize,offset=0)
        tax.ticks(axis='lbr', linewidth=1, multiple = 10, fontsize=5)
        cb_kwargs = {"shrink" : 2,
                     "orientation" : "horizontal",
                     "fraction" : 0.1,
                     "pad" : 0.05,
                     "aspect" : 30}
        self.plot_heatmap_ternary(data,tax,cb_kwargs=cb_kwargs)
        tax.scatter(
            [merged_mean],marker='x',s=10,linewidth=0.4,zorder=2,label=mean_label,
            c='red')
        colours=['orange','purple','green']
        for mean,label,colour in zip(small_means,point_labels,colours):
            tax.scatter(
                [mean],marker='x',linewidth=1.2,zorder=10,label=label,
                c=colour)
        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        ax=tax.get_axes()
        ax.legend(loc='upper right')
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

    def plot_heatmap_ternary(
                self,data,tax,scale=100,cb_kwargs=None,show=False):
            tax.heatmap(data,scale,cb_kwargs=cb_kwargs)
            if show:
                plt.show()
            return tax

    def quickplot(
            self,data,filename):
        fig,tax=ternary.figure(scale=100)
        tax.heatmap(data,100)
        plt.savefig(filename)
        plt.clf()

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
            
    def plot_points_berny(self,tax,points,labels):
        for n,key in enumerate(labels):
            print(points)
            print(points[labels[key][0]:labels[key][1]])
            tax.scatter(
                points[labels[key][0]:labels[key][1]],label=key,
                color=self.cmap(n+1),zorder=8,marker='x',linewidth=0.5
            )

    def plot_goal_berny(self,tax):
        tax.scatter(
            [self.goal],color=self.cmap(0),zorder=8,marker='x',label='Goal')

    def plot_points(
            self,show=True,use_labels=False,tax=None,labels=None,**kwargs):
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

    def plot_lines(self,starts,finishes,tax,labels,**kwargs):
        print(len(starts),len(finishes))
        for n,key in enumerate(labels):
            if key !='Next sample point':
                s=labels[key][0]
                f=labels[key][1]
                for start,finish in zip(starts[s:f],finishes[s:f]):
                    tax.line(start,finish,color=self.cmap(n+1),zorder=8,
                             **kwargs)
        return tax


    def show_labels(self):
       self.show_labels=True


    def set_projection(self,elev,azim):
        self.ax.view_init(elev=elev,azim=azim)
        


