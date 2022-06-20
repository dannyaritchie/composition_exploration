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
        tax.top_corner_label(self.top,fontsize=fontsize,offset=0.15)
        tax.right_corner_label(self.right,fontsize=fontsize,offset=0.05)
        tax.left_corner_label(self.left,fontsize=fontsize,offset=0.05)
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

    def mean_line_small(self,points,end_points,mean,small_means):
        fig, tax= ternary.figure(scale=100)
        self.tax_setup(tax,'line')
        tax.scatter(
            points,label='Initial',color='Blue',**self.scatter_ka)
        tax.scatter(
            [mean],label='K',color='Green',**self.scatter_ka)
        tax.line(mean,points[0],linestyle='--',linewidth=0.5,color='Green')
        tax.line(points[0],end_points[0],linewidth=0.5,color='Blue')
        labels=['LiBr','CaBr$_2$']
        cs=['red','orange']
        for point,label,c in zip(small_means,labels,cs):
            tax.scatter(
                [point],label=label,color=c,**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + "mean_line_small.png")
        #plt.show()

    def mean_line_small_old(self,points,end_points,mean,small_means,old):
        fig, tax= ternary.figure(scale=100)
        self.tax_setup(tax,'line')
        tax.scatter(
            points,label='Initial',color='Blue',**self.scatter_ka)
        tax.scatter(
            old,label='Other sampled',color='red',**self.scatter_ka)
        tax.scatter(
            [mean],label='K',color='Green',**self.scatter_ka)
        tax.line(mean,points[0],linestyle='--',linewidth=0.5,color='Green')
        tax.line(points[0],end_points[0],linewidth=0.5,color='Blue')
        labels=['LiBr','CaBr$_2$']
        for point,label in zip(small_means,labels):
            tax.scatter(
                [point],label=label,color='orange',**self.scatter_ka)
        self.set_aspect(fig,tax)
        plt.savefig(self.directory + "mean_line_small_old.png")
        #plt.show()

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

    def merged_ball(self,data,mean,show=True):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter(
            [mean],label='K',color='Green',**self.scatter_ka)
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat')
        plt.savefig(self.directory + "mergedball.png")
        if show:
            plt.show()

    def p_line(self,data,points,end_points,show=True):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        for i in range(len(points)):
            tax.line(points[i],end_points[i],linewidth=0.5,color='Blue')
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat')
        #plt.savefig(self.directory + "p mean initial.png")
        if show:
            plt.show()

    def p_goal(self,data,goal,show=True):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter([goal],color='blue',**self.scatter_ka)
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat')
        #plt.savefig(self.directory + "p mean initial.png")
        if show:
            plt.show()

    def p_mean_initial(self,data,mean,points,show=False):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter(
            [mean],label='K',color='Green',**self.scatter_ka)
        tax.scatter(
            points,label='Initial',color='Blue',**self.scatter_ka)
        tax.heatmap(data=data,scale=100,cmap='Reds',cb_kwargs=self.cb_kwargs)
        self.set_aspect(fig,tax,kind='heat')
        plt.savefig(self.directory + "p mean initial.png")
        if show:
            plt.show()

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

    def line_goal(self,point,end_point,goal):
        fig,tax=ternary.figure(scale=100)
        self.tax_setup(tax,'heat')
        tax.scatter([goal],label='goal',color='blue',**self.scatter_ka)
        tax.line(point,end_point,linewidth=0.5,color='red',zorder=7)
        plt.show()

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


        


