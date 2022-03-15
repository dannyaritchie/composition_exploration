import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats

class error_propagator:

    def __init__(self,dim=2):
        self.dim=dim

    def set_moles_error(self,moles,formulas,moles_error):
        self.moles=np.array(moles)
        self.moles_error=np.array(moles_error)
        self.small_balls_mean_unnorm=np.array(formulas)
        for i in self.small_balls_mean_unnorm:
            print('start mean',i)
        merged_mean=np.zeros(self.small_balls_mean_unnorm.shape[1])
        merged_mean_std=np.zeros(self.small_balls_mean_unnorm.shape[1])
        for i in range(len(moles)):
            merged_mean=merged_mean+self.moles[i]*self.small_balls_mean_unnorm[i]
            merged_mean_std=merged_mean_std+(self.moles_error[i]*self.small_balls_mean_unnorm[i])**2
        self.merged_mean=merged_mean/np.sum(merged_mean)
        self.merged_mean_std=merged_mean_std**0.5/np.sum(merged_mean)
        self.merged_sigma = np.diag(self.merged_mean_std)
        #set small ball means and error
        #errors are scaled so that when combine the balls are of right scale
        self.weighted_moles_error=[0]*len(moles_error)
        for i in range(len(moles)):
            weighted_mole_error=moles_error[i]/np.sum(merged_mean)
            self.weighted_moles_error[i]=weighted_mole_error
        self.small_balls_sigma=np.empty((len(formulas),self.dim,self.dim))
        print('hihi',self.small_balls_mean_unnorm.shape)
        self.small_balls_mean=np.empty((self.small_balls_mean_unnorm.shape))
        for i in range(len(self.small_balls_mean)):
            self.small_balls_mean[i]=self.small_balls_mean_unnorm[i]/np.sum(self.small_balls_mean_unnorm[i])
        for i in range(len(formulas)):
            diag=np.array([x*self.weighted_moles_error[i] for x in formulas[i]])
            self.small_balls_sigma[i] = np.diag(diag)

    def get_small_balls(self):
        return (self.small_balls_mean,self.small_balls_sigma)
            
    def get_merged_balls(self):
        return (self.merged_mean,self.merged_sigma)

    def get_merged_balls_p(self,us):
    #Todo leaving the plotting basis constant while changing the simulation
    #basis could result in a random error that helps quantify the error induced
    #by this simulation
        self.project_merged_ball_onto_us(us)
        return (self.merged_mean_p,self.merged_sigma_p)

    def project_merged_ball_onto_us(self,us):
        self.merged_mean_p = np.empty((2))
        self.merged_sigma_p = np.empty((2))
        merged_p=self.project_onto_us(us,self.merged_sigma,self.merged_mean)
        self.merged_mean_p=merged_p[0]
        self.merged_sigma_p=merged_p[1]

    def project_small_balls_onto_us(self,us):
        self.small_balls_mean_p = np.empty((self.small_balls_mean.shape[0],2))
        self.small_balls_sigma_p = np.empty((self.small_balls_sigma.shape[0],2))
        for n,(mean,sigma) in enumerate(zip(self.small_balls_mean,self.small_balls_sigma)):
            small_balls_p=self.project_onto_us(us,sigma,mean)
            self.small_balls_mean_p[n]=small_balls_p[0]
            self.small_balls_sigma_p[n]=small_balls_p[1]

    def get_small_balls_p(self,us):
        #returns mean and sigma (both dimesnion of us[1]) for each ball
        self.project_small_balls_onto_us(us)
        for i in self.small_balls_mean_p:
            print('N-1 ball mean',i)
        return (self.small_balls_mean_p,self.small_balls_sigma_p)

    def initialise(self,method,goal,sample):
        if method=='a':
            u0=goal-sample
            u0=u0/np.linalg.norm(u0)
            self.z=0.3
            self.set_mean(sample+self.z*u0)
            sigma=np.array([[0.1,0,0,0,0],
                           [0,0.1,0,0,0],
                           [0,0,0.1,0,0],
                           [0,0,0,0.1,0],
                           [0,0,0,0,0.1]])
            self.set_sigma(sigma)
        if method=='b':
            u0=goal-sample
            u0=u0/np.linalg.norm(u0)
            self.z=0.3
            self.set_mean(sample+self.z*u0)
            sigma=np.array([[0.01,0,0],
                            [0,0.01,0],
                            [0,0,0.01]])
            self.set_sigma(sigma)
    def set_mean(self,mean):
        if len(mean)==self.dim:
            self.mean=mean
        else:
            print('Error, mean of incorrect dimension')

    def set_sigma(self,sigma):
        if sigma.shape[0]==self.dim and sigma.shape[1]==self.dim:
            self.sigma = sigma

    def get_std(self,u):
        self.project_onto_u(u)
        return np.sqrt(self.projected_sigma/self.z)
            
    def test_2d(self):
        mean=np.array([1,1])
        sigma=np.array([[1,0],[0,1]])
        self.set_mean(mean)
        self.set_sigma(sigma)
        us=np.array([[1/np.sqrt(5),2/np.sqrt(5)]])
        self.project_onto_us(us)
        self.visualise_projected_2d()

    def project_onto_us(self,us,sigma=None,mean=None):
        if sigma is None:
            print('hey')
            sigma=self.sigma
            mean=self.mean
        if us.shape[1]!=self.dim:
            print('Error, projection vectors must have same dimension as space')
        else:
            print('pre proj mean',mean)
            self.us=us
            projected_means=np.matmul(us,mean)
            print('N-3 proj mean',projected_means)
            projected_sigmas=np.empty((len(us)))
            for n,u in enumerate(us):
                projected_sigma=np.array([np.einsum('k,kl,l',u,sigma,u)])
                projected_sigmas[n]=projected_sigma
            if sigma is None:
                self.projected_sigmas=projected_sigmas
                self.projected_means=projected_means
            else:
                return (projected_means,projected_sigmas)



    def project_onto_u(self,u):
        print(u.shape)
        if u.shape[0]!=self.dim:
            print('Error, projection vectors must have same dimension as space')
        else:
            self.u=u
            self.projected_mean=np.array([np.dot(self.mean,u)])
            self.projected_sigma=np.array([np.einsum('k,kl,l',u,self.sigma,u)])

    def visualise_og2d(self,N=50,xmin=-3,xmax=3,ymin=-3,ymax=3,ax=None):
        if self.dim!=2:
            print('Error, visualise only works for 2d')
        else:
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

            # Adjust the limits, ticks and view angle

            plt.show()
    def visualise_projected_2d(self,n=70):
        sigma=self.projected_sigmas[0]
        mean=self.projected_means[0]
        print('mean: ',mean,'should',3/np.sqrt(5))
        X=np.linspace(-3*np.sqrt(5)-2*np.sqrt(5)/5,3*np.sqrt(5)-2*np.sqrt(5)/5,n)
        Z=scipy.stats.norm(mean, np.sqrt(sigma)).pdf(X)
        print(Z.shape)
        print('poo',)
        t=np.broadcast_to(np.array([2/5,-1/5]),(n,self.dim))
        x=np.broadcast_to(X,(self.dim,n)).T*np.broadcast_to(self.us[0],(n,self.dim))+t
        for i in x:
            print(i)
        print(x.shape)
        f, (ax1,ax2)=plt.subplots(1,2)
        ax1.scatter(x[:,0],x[:,1],c=Z)
        ax1.scatter(mean*self.us[0][0]+2/5,mean*self.us[0][1]-1/5,c='black')
        ax1.set_ylim(-3,3)
        ax1.set_xlim(-3,3)
        self.visualise_og2d(ax=ax2)
        print(mean*self.us[0][0],mean*self.us[0][1])
        print(self.mean)
        plt.show()
