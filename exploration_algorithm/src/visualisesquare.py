import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class visualise_square:
    def generate_grid(self):
        x=np.empty((3))
    
    def plot_omega(self,omega,ax=None,show=False,cube_size=None):
        self.plot_scatter(ax,omega,show,cube_size=cube_size)

    def create_boundaries(self,A,contained_point):
        p1=np.array([0,0,0,100])
        p2=np.array([0,0,100,0])
        p3=np.array([0,100,0,0])
        p4=np.array([100,0,0,0])

        sp1=p1-contained_point
        sp1=np.einsum('ij,j',A,sp1)
        sp2=p2-contained_point
        sp2=np.einsum('ij,j',A,sp2)
        sp3=p3-contained_point
        sp3=np.einsum('ij,j',A,sp3)
        sp4=p4-contained_point
        sp4=np.einsum('ij,j',A,sp4)
        print('-----------')
        print('a',sp1)
        print(sp2)
        print(sp3)
        print(sp4)

        lines=[]
        lines.append([sp1,sp2])
        lines.append([sp1,sp3])
        lines.append([sp1,sp4])
        lines.append([sp2,sp3])
        lines.append([sp2,sp4])
        lines.append([sp3,sp4])
        return lines


    def add_boundary(self,ax,A,contained_point):
        lines=self.create_boundaries(A,contained_point)
        self.draw_lines(lines,ax,show=False)

    
    def plot_scatter_from_standard(self,ax,points_standard,show=True):
        xs=[None]*len(points_standard)
        ys=[None]*len(points_standard)
        for n,i in enumerate(points_standard):
            x=np.dot(self.X,i)
            y=np.dot(self.Y,i)
            xs[n]=x
            ys[n]=y
        ax.scatter(xs,ys,s=10)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        if show:
            plt.show()

    def plot_scatter(
        self,ax,points_2d,lim=False,show=False,cube_size=None,use_axs=None,
        **kwargs):
        if use_axs is not None:
            ax=self.axs[use_axs[0]][use_axs[1]]
        elif ax is None:
            fig,ax=plt.subplots(1,1)
        xs=points_2d[:,0]
        ys=points_2d[:,1]
        if cube_size is not None:
            xs=xs/cube_size
            ys=ys/cube_size
        ax.scatter(xs,ys,**kwargs)
        if lim:
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
        if show:
            plt.show()

    def set_means(self,means):
        self.means=means

    def set_sigmas(self,sigmas):
        self.sigmas=sigmas

    def plot_small_balls(self):
        for i,j in zip(self.means,self.sigmas):
            print(i)
            print(j)

    def draw_ball_fig(self, merged_ball, balls, omega, cube_size=None):
            fig, axs = plt.subplots(2,2)
            span = 1
            xmin = -span
            ymin = -span
            xmax = span
            ymax = span
            N=50
            self.plot_balls(balls,xmin,xmax,ymin,ymax,N,
                            ax=axs[0][0],show=False)
            self.plot_ball(merged_ball,xmin,xmax,ymin,ymax,N,
                           ax=axs[0][1],show=False)
            self.plot_omega(omega,axs[1][0],show=False,cube_size=cube_size)
            plt.show()

    def plot_balls(self,balls,xmin=-2,xmax=2,ymin=-2,ymax=2,
               N=100,ax=None,show=True):
            X = np.linspace(xmin, xmax, N)
            Y = np.linspace(ymin, ymax, N)
            X, Y = np.meshgrid(X, Y)
            pos=np.empty(X.shape+(2,))
            pos[:,:,0]=X
            pos[:,:,1]=Y
            z=np.zeros((N,N))
            for mean,sigmas in zip(balls[0],balls[1]):
                print('N ball mean',mean)
                sigma=np.diag(sigmas)
                n = mean.shape[0]
                Sigma_det = np.linalg.det(sigma)
                Sigma_inv = np.linalg.inv(sigma)
                N = np.sqrt((2*np.pi)**n * Sigma_det)
                # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
                # way across all the input variables.
                fac = np.einsum('...k,kl,...l->...', pos-mean, Sigma_inv,
                                pos-mean)
                tZ = np.exp(-fac / 2) / N
                z = z + tZ
            if ax is None:
                fig,ax=plt.subplots(1,1)
            cset = ax.contourf(X, Y, z, cmap=cm.viridis)
            if show:
                plt.show()

        #takes ball in representation set of means, set of sigmas

    def plot_ball(self,ball,xmin,xmax,ymin,ymax,N,ax,show=True):
        print(ball[0])
        print(ball[1])
        self.sigma=np.diag(ball[1])
        self.mean=ball[0]
        self.draw_ball(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,N=N,
                      ax=ax)
        if show:
            plt.show()


    def draw_ball(self,N=50,xmin=-3,xmax=3,ymin=-3,ymax=3,ax=None):
            X = np.linspace(xmin, xmax, N)
            Y = np.linspace(ymin, ymax, N)
            X, Y = np.meshgrid(X, Y)
            pos=np.empty(X.shape+(2,))
            pos[:,:,0]=X
            pos[:,:,1]=Y
            n = self.mean.shape[0]
            Sigma_det = np.linalg.det(self.sigma)
            Sigma_inv = np.linalg.inv(self.sigma)
            N = np.sqrt((2*np.pi)**n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos-self.mean, Sigma_inv,
                            pos-self.mean)

            Z = np.exp(-fac / 2) / N
            # Create a surface plot and projected filled contour plot under it.
            if ax is None:
                fig = plt.figure()
                ax = plt.gca()
            #ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                            #cmap=cm.viridis)

                cset = ax.contourf(X, Y, Z, cmap=cm.viridis)
            else:
                ax.contourf(X,Y,Z,cmap=cm.viridis)

    def draw_lines(self,lines,ax,show=True,**kwargs):
        for line in lines:
            #print(line)
            ax.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],**kwargs)
        if show:
            plt.show()

    def test_fig(self,goal,points,lines,heatmap,xlim,ylim,omega,mean):
        f,ax=plt.subplots(1,2)
        self.goal_fig(goal,points,lines,ax=ax[0],show=False,lims=[xlim,ylim])
        self.plot_scatter(ax[0],np.array([mean]),show=False,lim=False,marker='x',c='green',label='mean')
        self.plot_heatmap(heatmap,xlim[0],xlim[1],ylim[0],ylim[1],ax=ax[1],show=False)
        plt.show()

    def test_heat(self,d1,d2,d3,xlim,ylim):
        f,ax=plt.subplots(1,3)
        self.plot_heatmap(d1,xlim[0],xlim[1],ylim[0],ylim[1],ax=ax[0],show=False)
        self.plot_heatmap(d2,xlim[0],xlim[1],ylim[0],ylim[1],ax=ax[1],show=False)
        self.plot_heatmap(d3,xlim[0],xlim[1],ylim[0],ylim[1],ax=ax[2],show=False)
        plt.show()

    def test_heatb(self,heatmap,xlim,ylim):
        f,ax=plt.subplots(1,1)
        self.plot_heatmap(heatmap,xlim[0],xlim[1],ylim[0],ylim[1],ax=ax,show=False)
        plt.show()
        
    def process_fig(self):
        self.f,self.axs=plt.subplots(2,3)

    def goal_fig(self,goal,points,lines,mean,ax=None,show=True,lims=None):
        if ax is None:
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
        print(goal)
        self.plot_scatter(
            ax,np.array([goal]),show=False,lim=False,marker='x',c='red',
            label='Goal')
        self.plot_scatter(
            ax,np.array([mean]),show=False,lim=False,marker='x',c='orange',
            label='mean')
        self.plot_scatter(ax,points,show=False,lim=None,label='Sample point')
        self.draw_lines(lines,ax,show=False)
        ax.legend()
        if lims is not None:
            ax.set_xlim(lims[0][0],lims[0][1])
            ax.set_ylim(lims[1][0],lims[1][1])
        if show:
            plt.show()

    def plot_heatmap(
        self,heatmap,xmin,xmax,ymin,ymax,ax=None,show=True,use_axs=None,
        filename=None,**kwargs):
        if use_axs is not None:
            ax=self.axs[use_axs[0]][use_axs[1]]
        elif ax is None:
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
        extent=(xmin,xmax,ymin,ymax)
        ax.imshow(heatmap,origin='lower',extent=extent)
        #ax.set_xlim([xlim[0],xlim[1]])
        #ax.set_ylim([ylim[0],ylim[1]])
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename)

    def next_sample_fig(self,heatmap,xlim,ylim,points,lines,goal,filename):
        fig,ax=plt.subplots(1,2)
        self.plot_heatmap(
            heatmap,xlim[0],xlim[1],ylim[0],ylim[1],show=False,ax=ax[0])
        self.plot_scatter(
            ax[1],np.array([goal]),show=False,marker='x',c='red',label='Goal')
        self.plot_scatter(
            ax[1],points,show=False,label='Initial point',c='orange')
        self.draw_lines(lines,ax[1],show=False)
        plt.savefig(filename)

    def estimated_known(self,closest,goal,est_known,A,contained_point,purity):
        fig, ax=plt.subplots(1,1)
        #ax.autoscale()
        self.add_boundary(ax,A,contained_point)
        print([closest[0]])
        #ax.scatter(closest[0],closest[1],label='closest',color='red')
        #ax.scatter(goal[0],goal[1],label='goal',color='green')
        #ax.scatter(est_known[0],est_known[1],label='est_known',color='blue')
        print('c:',closest)
        print(goal)
        print(est_known)
        print('------------')
        ax.legend()
        ax.set_title(str(purity)+'%')

        plt.show()

