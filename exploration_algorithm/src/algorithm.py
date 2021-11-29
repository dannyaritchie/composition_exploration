import ternary
import numpy as np
import numpy.ma as ma
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares, minimize, Bounds
import scipy.stats
from ternary.helpers import simplex_iterator
import random
import numba
from numba import njit
import time

@njit(parallel=True)
def expected_gain_in_info_numba_recieve(f,n,m,l,o,a,b,x,k):
        f_broad = np.broadcast_to(f,(m,n,l))
        #ab
        a_broad=np.broadcast_to(a,(m,n,o))
        b_broad=np.swapaxes(np.broadcast_to(b,(n,m,o)),0,1)
        ab = b_broad-a_broad
        ab_norm = np.linalg.norm(ab,axis=2)
        ab_norm_broad = np.moveaxis(np.broadcast_to(ab_norm,(o,m,n)),0,2)
        ab_normed = ab/ab_norm_broad
        ab_normed_broad = np.swapaxes(np.broadcast_to(ab_normed,(l,m,n,o)),0,1)
        #ax
        a_broad=np.broadcast_to(a,(l,n,o))
        x_broad=np.swapaxes(np.broadcast_to(x,(n,l,o)),0,1)
        ax = x_broad-a_broad
        ax_broad = np.broadcast_to(ax,(m,l,n,o))
        ax_norm = np.linalg.norm(ax,axis=2)
        ax_norm_broad = np.broadcast_to(ax_norm,(m,l,n))
        dot = np.ones((m,l,n))
        np.divide(np.sum(ax_broad*ab_normed_broad,axis=3),ax_norm_broad,out=dot,where=ax_norm_broad!=0)
        dot=np.clip(dot,-1,1)
        angle=np.arccos(dot)
        info=np.exp(-1*angle/k)
        info=np.swapaxes(info,1,2) # mnl
        info=info/np.moveaxis(np.broadcast_to(np.sum(info,axis=2),(l,m,n)),0,2)
        post_f = f_broad*info
        post_f=post_f/np.moveaxis(np.broadcast_to(np.sum(post_f,axis=2),(l,m,n)),0,2)
        print(expected_gain_in_info_numba_recieve.parallel_diagnostics(level=4))
        return post_f

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_angle(a,b,x):
    #gets the angle between the line ab and the line ax 
    ax = numpy.subtract(x,a)
    ab = numpy.subtract(b,a)
    theta = math.acos(np.dot(ax,ab)/(np.linalg.norm(ax)*np.linalg.norm(ab)))

class all_information:

    def __init__(self,dim=3):
        self.lines = np.empty((0,dim))
        #direction vector of lines
        #starting point of lines
        self.end_points = np.empty((0,dim))
        self.labels = None
        self.optimal_point=None

    def getdistance(self):
        return np.sqrt(np.sum((self.goal-self.optimal_point)**2))

    def add_point(self,point):
        self.points = np.append(self.points,point,axis=0)

    def sampleonline(self, b, line_index):
        min_steps=9999
        axis_hit=None
        line = self.lines[line_index]
        point = self.points[line_index]
        for i in range(len(line)):
            if line[i] < 0:
                if point[i]/(-1*line[i]) < min_steps:
                    min_steps = -1*point[i]/line[i]
                    axis_hit=i
        x_i = point[axis_hit]
        c = -x_i/line[axis_hit]
        start = len(self.points)
        for i in range(1,b+1):
            #target_point = [x+i*c*y/(batchsize+1) for x,y in zip(point,line)]
            self.points =  np.append(self.points,np.array([np.add(point,i*c*line/(b))]),axis=0)
        end = len(self.points)
        return [start,end]

    def set_labels(self,method='secondary',batch_size=0):
        if method == 'secondary':
            self.labels = {'initial':[0,1], 'secondary':[1,len(self.lines)]}
        if method == 'random':
            self.labels = {'random':[0,len(self.points)]}
        if method == '1_batch':
            self.labels = {'initial':[0,1],
                           'primary':[1,len(self.lines)-batch_size],
                           'secondary':[len(self.lines)-batch_size,len(self.lines)]}



    def create_line_from_point(self,point_index,angle):
        point = self.points[point_index]
        if not (np.array_equal(point,[100,0,0])):
            a = np.subtract(np.array([100,0,0]),point)
        else:
            a = np.subtract(np.array([0,0,100]),point)
        if not (np.array_equal(point,[0,100,0])):
            b = np.subtract(np.array([0,100,0]),point)
        else:
            b = np.subtract(np.array([0,0,100]),point)
        rotation_axis = np.cross(a,b)
        if (np.linalg.norm(rotation_axis) == 0):
            b = np.subtract(np.array([25,25,50]),point)
        rotation_axis = np.cross(a,b)
        rotation_vec = rotation_axis * angle / np.linalg.norm(rotation_axis)
        rotation = r.from_rotvec(rotation_vec)
        summ = 0
        sumx = 0
        sumy = 0
        line = np.subtract(self.goal,point)
        rotated_line = rotation.apply(line)
        self.lines = np.append(self.lines,[rotated_line],axis=0)

    def set_end_points(self):
        #to do - np vectorise
        self.end_points = np.empty((0,3))
        for point,line in zip(self.points,self.lines):
            min_steps=9999
            for pi,vi in zip(point,line):
                if vi < 0:
                    if pi/(-1*vi) < min_steps:
                        mifn_steps = -1*pi/vi
            final_point = np.add(point,line*min_steps)
            ab = final_point-point
            printid = True
            for i in ab:
                if i != 0:
                    printid = False
            if printid:
                print('here',ab)
                print(line)
                print(point)
                print(min_steps,'m')
            self.end_points = np.append(self.end_points,[final_point],axis=0)

    def plot(self,goal,points,lines,
             optimal_point = False,angle_product_heatmap = False,
             title = 'heatmap', show=True, save=False, scale=100,
             name = 'test.png',compute_theta=True ,tax=None):
        if tax is None:
            figure, tax = ternary.figure(scale=scale)
        tax.set_title(title,fontsize = 10)
        number_colours = 0
        subplots = False
        if (lines):
            number_colours += len(self.lines)
        if goal:
            number_colours += 1
        if  points:
            if self.labels:
                number_colours+=len(self.labels)
        if optimal_point:
            number_colours +=1
        if angle_product_heatmap:
            plt.clf()
            f, (ax1, ax2) = plt.subplots(1,2)
            scale = 100
            figure, tax = ternary.figure(ax = ax1, scale = scale)
            #now you can use ternary normally:
            self.plot_theta_score(tax,compute=compute_theta)
            subplots = True
            figure, tax = ternary.figure(ax = ax2, scale = scale)
            '''
            fig1, f1_axes = plt.subplots(ncols=2, nrows=1, constrained_layout=true)
            taxa = ternary.ternaryaxessubplot(ax=f1_axes[0])
            tax = ternary.ternaryaxessubplot(ax=f1_axes[1])
            self.plot_theta_score(taxa)
            subplots = true
            '''
        cmap = get_cmap(number_colours)
        used_colours = 0
        if goal:
            self.plot_goal(tax,cmap(used_colours))
            used_colours += 1
        if optimal_point:
            self.plot_optimal_point(tax,cmap(used_colours))
            used_colours += 1
        if points:
            self.plot_points(tax,cmap,used_colours)
        if lines:
            self.plot_lines(tax,cmap,used_colours)
        tax.clear_matplotlib_ticks()
        tax.boundary(linewidth=1)
        tax.gridlines(color="black", multiple=10)
        tax.ticks(axis='lbr', linewidth=1, multiple = 10, fontsize=5)
        ax=tax.get_axes()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if subplots:
            f.set_size_inches(25,10)
            f.suptitle(title,fontsize = 10)
        if save:
            plt.savefig(name,dpi=800)
        if show:
            plt.show()
        return tax
        plt.clf()

    def plot_lines(self,tax,cmap,used_colours):
        c_edge_intersect = np.empty((0,self.lines.shape[1]))
        for point,line in zip(self.points,self.lines):
            min_steps=9999
            for pi,vi in zip(point,line):
                if vi < 0:
                    if pi/(-1*vi) < min_steps:
                        min_steps = -1*pi/vi
            final_point = np.add(point,line*min_steps)
            c_edge_intersect = np.append(c_edge_intersect,[final_point],axis=0)

        for n,i in enumerate(self.lines):
            tax.line(self.points[n], c_edge_intersect[n], linewidth=1.,
                     color=cmap(n+used_colours), linestyle="-",label=str(n))
        return tax

    def plot_points(self,tax,cmap,used_colours):
        for n,key in enumerate(self.labels):
            tax.scatter(self.points[self.labels[key][0]:self.labels[key][1]],
                        marker='x',s=50,linewidth = 1.,
                        color=cmap(used_colours+n),
                        label = key)
        return tax

    def plot_optimal_point(self,tax,colour):
        tax.scatter([self.optimal_point], marker='x', s=50,
                    color=colour,label='optimal')
        return tax

    def plot_goal(self,tax,colour):
        tax.scatter([self.goal], marker='x', s=50, color=colour,label='target')
        return tax

    def least_squares_solution(self):
        sum_a = np.zeros((3,3))
        sum_p = np.zeros((3))
        for line, point in zip(self.lines,self.points):
            line = line/np.linalg.norm(line)
            a = np.identity(3) - np.outer(line,line)
            p = np.matmul(np.identity(3) - np.outer(line,line),point)
            sum_a += a
            sum_p += p
        solution = np.linalg.lstsq(sum_a,sum_p,rcond=none)
        tax = self.plot(true,true,true,false,show=false)
        tax.scatter([solution[0]],color='pink',label='least squares solution')
        ax=tax.get_axes()
        # shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("presentation/a/leastsquares.eps",dpi=1200)

    def angle(self,x):
        #returns the n array where the i'th element is the angle between the
        #ax and the vector ab for the i'th line in self.lines going from a to b
        ax=np.subtract(x,self.points)
        ab=np.subtract(self.end_points,self.points)
        for n,i in enumerate(np.linalg.norm(ax,axis=1)):
            if i ==0:
                print(ax[n],'= ax with 0 normal')
        u = (ax*ab).sum(axis=1)/(np.linalg.norm(ax,axis=1)*np.linalg.norm(ab,axis=1))
        for n,i in enumerate(u):
            if i > 1:
                u[n] = 1
        return np.arccos(u)

    def jacobian_angle(self,x):
        #returns the n,3 array where the i,j element is the differential of the
        #i'th with respect to the j'th axis
        ax=np.subtract(x,self.points)
        ab=np.subtract(self.end_points,self.points)
        ax_norm = np.linalg.norm(ax,axis=1)
        ab_norm = np.linalg.norm(ab,axis=1)
        u = np.sum(ax*ab,axis=1)/(ax_norm*ab_norm)
        #  c = -1/(np.sqrt(1-u**2))
        ab_c = 1/(ax_norm*ab_norm)
        ax_c = u/(ax_norm**2)
        m = len(self.lines)
        ab_c=np.transpose(np.broadcast_to(ab_c,(3,m)))
        ax_c=np.transpose(np.broadcast_to(ax_c,(3,m)))
        # c=np.transpose(np.broadcast_to(c,(3,m))) implementdd in other function
        # return c*(ab_c*ab-ax_c*ax)
        return (ab_c*ab-ax_c*ax)

    def nllsq_angle(self):
        print(sum(least_squares(self.angle,np.array([101/3,100/3,100/3]),jac=self.jacobian_angle,method='lm',xtol=0.1,).x))

    def angle_product(self,x,k=np.pi):
        angle = self.angle(x)
        return 1-np.prod(np.exp(-1*angle/k))

    def cgmo_angle(self, method='SLSQP'):
        x0=np.array([100/3,100/3,100/3])
        bounds = Bounds([0]*len(x0),[100]*len(x0))
        eq_cons = {'type' : 'eq',
                   'fun' : lambda x: np.array([np.sum(x)-100]),
                   'jac' : lambda x: np.array([1]*len(x))}
        res = minimize(self.angle_product,x0,method=method,
                        jac=self.angle_prod_jac,constraints=[eq_cons],
                       bounds=bounds,options={'ftol': 1e-20, 'maxiter':5000})
        self.optimal_point = res.x
        print(res)
        return res

    def angle_prod_jac(self,x,k=np.pi):
        m = len(self.lines)
        a = self.angle_product(x)
        grad_u = self.jacobian_angle(x)
        ax=np.subtract(x,self.points)
        ab=np.subtract(self.end_points,self.points)
        u =(ax*ab).sum(axis=1)/(np.linalg.norm(ax,axis=1)*np.linalg.norm(ab,axis=1))
        for n,i in enumerate(u):
            if i >=1-1e-9:
                u[n] = 1-1e-9
        b = 1/np.sqrt(1-u**2)
        b_broad = np.swapaxes(np.broadcast_to(b,(3,m)),0,1)
        bc = b_broad*grad_u
        e = np.sum(bc,axis=0)
        gradient = (1/k) * e * a
        return -1*gradient

    def jacobian_angle_product(self,x):
        #for i in len(x) retuurns some complicated maths
        F = self.angle(x)
        result = np.empty(shape=(len(F)))
        for i in range(len(F)):
            mask = [False]*len(F)
            mask[i] = True
            result[i]=np.prod(ma.masked_array(F,mask=mask))
        jac_F = self.jacobian_angle(x)
        m = len(self.lines)
        presum = jac_F * np.transpose(np.broadcast_to(result,(3,m)))
        jacobian = np.sum(presum,axis=0)
        return jacobian

    def sum_theta_score(self,p):
        #sets the sum theta score
        #To do - vectorise
        P = [x*100 for x in p]
        total = 0
        for point,line in zip(self.points,self.lines):
            dot=np.dot(line, [x-y for x,y in zip(P,point)])/np.linalg.norm(line)
            if dot < 0:
                total += np.pi/2;
            else:
                cross = np.cross([x-y for x,y in
                                  zip(P,point)],line)/np.linalg.norm(line)
                total += np.arctan(np.linalg.norm(cross)/dot)
        return total

    def prod_theta_score(self,k=np.pi):
        #redundant?
        increment=1
        n = (100+increment)/increment
        length = int(n*(n+1)/2)
        omega = np.empty((length,3))
        keys = [None]*length
        count = 0
        m = len(self.points)
        for i in range(0,100+increment,increment):
            for j in range(0,100+increment-i,increment):
                x = np.array([i,j,100-i-j])
                omega[count] = x
                keys[count] = (i,j)
                count +=1
        a = self.points
        b = self.end_points
        a_broad = np.broadcast_to(a,(count,m,3))
        a_broad_t = np.swapaxes(a_broad,0,1)
        omega_broad = np.broadcast_to(omega,(m,count,3))
        a_omega = omega_broad-a_broad_t
        a_omega_normed = a_omega/np.moveaxis(np.broadcast_to(np.linalg.norm(a_omega,axis=2),(3,m,count)),0,2)
        ab=np.subtract(self.end_points,self.points)
        ab_norm = np.linalg.norm(ab,axis=1)
        ab_norm_t = np.swapaxes(np.broadcast_to(ab_norm,(3,m)),0,1)
        ab_normed = ab/ab_norm_t
        ab_normed_broad = np.swapaxes(np.broadcast_to(ab_normed,(count,m,3)),0,1)
        dot = (ab_normed_broad*a_omega_normed).sum(axis=2)
        dot = np.clip(dot,-1,1)
        theta = np.arccos(dot)
        theta_p = np.exp(-1*theta/k)
        values = np.prod(theta_p,axis=0)
        data = dict(zip(keys,values))
        return data

    def create_omega(self,increment,dim,timed=False):
        #todo - allow for increments smaller than 1 as well as arbitary regions
        t0 = time.time()
        n = int((100+increment)/increment)
        if dim == 4:
            total = 0
            for i in range(0,n):
                total += int((n-i)*(n-i+1)/2)
            omega = np.empty((total,4))
            count = 0
            for i in range(0,100+increment,increment):
                for j in range(0,100+increment-i,increment):
                    for k in range(0,100+increment-i-j,increment):
                        omega[count]=np.array([i,j,k,100-i-j-k])
                        count = count + 1
            t1 = time.time()
            if timed:
                print('omega make time: ',t1-t0,', increment: ',increment, ', dim: ',dim)
        if dim == 3:
            total = int(n*(n+1)/2)
            omega = np.empty((total,3))
            count = 0
            keys = [None]*total
            for i in range(0,100+increment,increment):
                for j in range(0,100+increment-i,increment):
                    omega[count]=np.array([i,j,100-i-j])
                    keys[count]=(i,j)
                    count = count + 1
            t1 = time.time()
            if timed:
                print('omega make time: ',t1-t0,', increment: ',increment, ', dim: ',dim)
            self.plotting_indicis = keys
        return omega

    def prod_theta_score_n(self,k=np.pi,dim=4,increment=10,
                           return_type = None, timed=False,
                          a=None,b=None):
        #Optimise so that End points MUST be 1 away from start points
        #block 1
        if a is None:
            self.omega = self.create_omega(increment,dim)
            a = self.points
            b = self.end_points
        omega=self.omega
        count = len(omega)
        m = len(b) #nimber of lines
        #block 2
        t1 = time.time()
        #broadcast a to repeated count times in its middle direction
        #(start point has no dependance on x)
        a_broad = np.broadcast_to(a,(count,m,dim))
        a_broad_t = np.swapaxes(a_broad,0,1)
        #broadcast omega to be repeated m times on axis 0,
        # (need 1 'omega' for every line
        omega_broad = np.broadcast_to(omega,(m,count,dim))
        #get the distance between a and x for every x
        a_omega = omega_broad-a_broad_t
        a_omega_norm = np.linalg.norm(a_omega,axis=2)
        #normalise so that 2-norm of axis 2 is 1
        #a_omega_normed = a_omega/np.moveaxis(
        #    np.broadcast_to(np.linalg.norm(a_omega,axis=2),(dim,m,count)),0,2)
        #get the vector from a to b (already contatined in self.lines????
        ab=np.subtract(b,a)
        #normalise
        ab_norm = np.linalg.norm(ab,axis=1)
        ab_norm_t = np.swapaxes(np.broadcast_to(ab_norm,(dim,m)),0,1)
        ab_normed = ab/ab_norm_t
        #broadcast to m by count by dim
        ab_normed_broad = np.swapaxes(np.broadcast_to(ab_normed,(count,m,dim)),0,1)
        dot=np.ones((m,count))
        dot = np.divide((ab_normed_broad*a_omega).sum(axis=2),
                        a_omega_norm,out=dot,where=a_omega_norm!=0)
        dot = np.clip(dot,-1,1)
        theta = np.arccos(dot)
        theta_p = np.exp(-1*theta/k)
        #get the count long vector representing value for each point in omega
        #by taking product of lines for each x in omega
        theta_p=theta_p/np.swapaxes(np.broadcast_to(np.sum(theta_p,axis=1),(count,m)),0,1)
        values = np.prod(theta_p,axis=0)
        #normalise
        #values=values/np.sum(values)
        if return_type == 'with_index':
            index=np.arange(0,len(values),step=1,dtype='int_')
            values_index = np.stack((values,index),axis=1)
            self.values = values_index
        elif return_type == 'to_include':
            return values
        else:
            print('error')
            self.values = values
        #block 3
        t2 = time.time()
        if timed:
            print('Block 2: ', t2-t1)

    def get_points_targets_for_exploration_evaluation(self, method='inc_exploration',
        a = 0.2, num_points = 10, num_targets = 100):
    #implementation notes
    #could write a ufunc to mask all values over a certain amount
    #could sort then take some chunk
    #ive gone with sort methid but it might not be faster
        if method == 'random_from_top_frac':
            #current doesnt work as nothing stopping and b from being the same
            #point
            sorted_values = self.values[self.values[:,0].argsort()]
            n = int(round(len(sorted_values)/a))
            viable_points = sorted_values[-n:]
            p = viable_points[:,0]/np.sum(viable_points[:,0])
            indexes = np.random.choice(viable_points[:,1],size=num_points,p=p).astype('int_')
            self.exploration_points = self.omega[indexes]
           # for i,j in zip(sorted_values[1],self.values[1]):
            #    print(i,'----',j)
        if method == 'inc_exploration':
            sorted_values = self.values[self.values[:,0].argsort()]
            n = int(round(len(sorted_values)*a))
            viable_points = sorted_values[-n:]
            p = viable_points[:,0]/np.sum(viable_points[:,0])
            indexes = np.random.choice(viable_points[:,1],size=num_points+num_targets,replace=False,p=p).astype('int_')
            self.exploration_points = self.omega[indexes][:num_points]
            self.exploration_targets = self.omega[indexes][num_points:]

    def weight_points_for_exploration_evaluation(self,method='angular_equivalence_with_cutoff',cutoff = 10):
        l = len(self.values)
        n = len(self.exploration_points)
        o = self.omega.shape[1]
        pretend_goals = self.exploration_targets
        m = len(pretend_goals)
        #ax 
        a_broad = np.swapaxes(np.broadcast_to(self.exploration_points,(l,n,o)),0,1)
        x_broad = np.broadcast_to(self.omega,(n,l,o))
        ax = x_broad-a_broad
        #ax norm
        ax_norms = np.linalg.norm(ax,axis=2)
        #ab = 
        a_broad = np.swapaxes(np.broadcast_to(self.exploration_points,(m,n,o)),0,1)
        b_broad = np.broadcast_to(pretend_goals,(n,m,o))
        ab = b_broad - a_broad
        #ab norms
        ab_norms = np.linalg.norm(ab,axis=2)
        if np.any(ab_norms == 0):
            print('Error, ab normal == 0')
        #ax dot ab
        ab_broad=np.swapaxes(np.broadcast_to(ab,(l,n,m,o)),0,2)
        ax_broad=np.broadcast_to(ax,(m,n,l,o))
        ab_norms_broad=np.swapaxes(np.broadcast_to(ab_norms,(l,n,m)),0,2)
        ax_norms_broad=np.broadcast_to(ax_norms,(m,n,l))
        norms = ab_norms_broad*ax_norms_broad
        #if ax norms == 0 (x and a are the same point)
        #then set the result equal to 1, so arrcos returns 0 and its not masked
        dot=np.ones((m,n,l))
        np.divide(np.sum(ab_broad*ax_broad,axis=3),norms,out=dot,where=norms!=0)
        dot = np.clip(dot,-1,1)
        angle=np.arccos(dot)
        masked_angle = ma.masked_greater(angle,np.pi*cutoff/180)
        f = self.values[:,0]/np.sum(self.values[:,0])
        f_broad = np.broadcast_to(f,(m,n,l))
        f_broad_masked = ma.masked_array(f_broad, mask = masked_angle.mask)
        weights = f_broad_masked.sum(axis=2)
        weights=weights/np.sum(weights,axis=0)
        return weights

    def expected_gain_in_info(self,k=np.pi):
        f = self.values[:,0]
        n = len(self.exploration_points)
        m = len(self.exploration_targets)
        l = len(f)
        o = self.omega.shape[1]
        f_broad = np.broadcast_to(f,(m,n,l))
        #ab
        a_broad=np.broadcast_to(self.exploration_points,(m,n,o))
        b_broad=np.swapaxes(np.broadcast_to(self.exploration_targets,(n,m,o)),0,1)
        ab = b_broad-a_broad
        ab_norm = np.linalg.norm(ab,axis=2)
        ab_norm_broad = np.moveaxis(np.broadcast_to(ab_norm,(o,m,n)),0,2)
        ab_normed = ab/ab_norm_broad
        ab_normed_broad = np.swapaxes(np.broadcast_to(ab_normed,(l,m,n,o)),0,1)
        #ax
        a_broad=np.broadcast_to(self.exploration_points,(l,n,o))
        x_broad=np.swapaxes(np.broadcast_to(self.omega,(n,l,o)),0,1)
        ax = x_broad-a_broad
        ax_broad = np.broadcast_to(ax,(m,l,n,o))
        ax_norm = np.linalg.norm(ax,axis=2)
        ax_norm_broad = np.broadcast_to(ax_norm,(m,l,n))
        dot = np.ones((m,l,n))
        np.divide(np.sum(ax_broad*ab_normed_broad,axis=3),ax_norm_broad,out=dot,where=ax_norm_broad!=0)
        dot=np.clip(dot,-1,1)
        angle=np.arccos(dot)
        info=np.exp(-1*angle/k)
        info=np.swapaxes(info,1,2) # mnl
        info=info/np.moveaxis(np.broadcast_to(np.sum(info,axis=2),(l,m,n)),0,2)
        original_score = scipy.stats.skew(np.sort(f))
        post_f = f_broad*info
        post_f=post_f/np.moveaxis(np.broadcast_to(np.sum(post_f,axis=2),(l,m,n)),0,2)
        post_score=scipy.stats.skew(np.sort(post_f,axis=2),axis=2)
        change_score=original_score-post_score
        return change_score



    def expected_gain_in_info_numba_send(self,k=np.pi):
        f = self.values[:,0]
        n = len(self.exploration_points)
        m = len(self.exploration_targets)
        l = len(f)
        o = self.omega.shape[1]
        a = self.exploration_points
        b = self.exploration_targets
        x = self.omega
        post_f=expected_gain_in_info_numba_recieve(f,n,m,l,o,a,b,x,k)
        print('a')
        print(expected_gain_in_info_numba_recieve.parallel_diagnostics(level=4))
        original_score = scipy.stats.skew(np.sort(f))
        post_score=scipy.stats.skew(np.sort(post_f,axis=2),axis=2)
        change_score=original_score-post_score
        return change_score

    def evaluation_test(self,dim=3,num_points=5,theta=20,theta_distribution='uniform', angular_equivalence=20,increment=1):
        self.add_random_initial(dim)
        self.add_random_goal(dim)
        self.lines = np.empty((0,dim))
        self.end_points = np.empty((0,dim))
        for i in range(num_points-1):
            self.points=np.append(self.points,np.array([self.random_point(dim)]),axis=0)
        for i in range(num_points):
            self.create_line_from_point_nd(dim,i,theta,theta_distribution)
        self.prod_theta_score_n(dim=dim,increment=increment,k=np.pi,return_type='with_index',create_grid=True)
        self.get_points_targets_for_exploration_evaluation(a=0.2,num_points=10,num_targets=100)
        weights = self.weight_points_for_exploration_evaluation(cutoff = angular_equivalence)
        change_score = self.expected_gain_in_info_numba_send()
        expected_score=np.sum(weights*change_score,axis=0)
        sorted_trial_points=self.exploration_points[expected_score.argsort()]
        self.points=np.append(self.points,np.array([sorted_trial_points[-1]]))
        self.create_line_from_point_nd(dim,len(self.points)-1,theta,theta_distribution)
        values_extra=self.prod_theta_score_n(dim=dim,increment=increment,k=np.pi,return_type='to_include',create_grid=True,a=np.array([self.points[-1]]),b=np.array([self.end_points[-1]]))
        self.values[:,0]=self.values[:,0]*values_extra

    def usual_setup(self,dim,theta_range,theta_distribution,batch_size):
        self.add_random_initial(dim)
        self.add_random_goal(dim)
        self.lines = np.empty((0,dim))
        self.end_points = np.empty((0,dim))
        self.create_line_from_point_nd(dim, 0, theta_range, theta_distribution)
        target_points = self.sampleonline(batch_size,0)
        for i in range(target_points[0],target_points[1]):
            self.create_line_from_point_nd(dim,i,theta_range,theta_distribution)


    def evaluation(self,dim,num_points,num_targets,theta_range,theta_distribution, angular_equivalence,increment,considered_fraction,batch_size,k,plot='',plot_process=False):
        if plot_process:
            if dim != 3:
                print('Error, cant plot for dim != 3')
                return
        self.usual_setup(dim,theta_range,theta_distribution,batch_size)
        f, axn = (None,None)
        if plot_process:
            f, axn = plt.subplots(2,3)
            f.set_figheight(15)
            f.set_figwidth(30)
        t0 = time.time()
        self.prod_theta_score_n(dim=dim,increment=increment,k=np.pi,return_type='with_index')
        values=self.values
        t1 = time.time()
        for i in range(batch_size):
            t15 = time.time()
            ax = None
            if plot_process:
                if i < 2:
                    ax = axn[0][1+i]
                else:
                    ax = axn[1][2-i]
            t2 = time.time()
            print('Time nil', t2-t15)
            self.get_points_targets_for_exploration_evaluation(a=considered_fraction,num_points=num_points,num_targets=num_targets)
            t3 = time.time()
            print('Time gp', t3-t2)
            weights=self.weight_points_for_exploration_evaluation(cutoff = angular_equivalence)
            t35 = time.time()
            change_score = self.expected_gain_in_info()
            t37 = time.time()
            expected_score=np.sum(weights*change_score,axis=0)
            sorted_trial_points=self.exploration_points[expected_score.argsort()]
            if plot_process:
                pl=['Trial Point','Best Point']
                p=[sorted_trial_points[:-1],[sorted_trial_points[-1]]]
                figure, tax = ternary.figure(ax = ax, scale = 100)
                tax=self.plot_theta_score(tax,compute=False,points=p,
                    point_labels=pl)
            self.points=np.append(self.points,np.array([sorted_trial_points[-1]]),axis=0)
            self.create_false_line(len(self.lines))
            values_extra=self.prod_theta_score_n(dim=dim,increment=increment,k=np.pi,return_type='to_include',a=np.array([self.points[-1]]),b=np.array([self.end_points[-1]]))
            t4 = time.time()
            print('Time wp', t35-t3)
            print('Time EXG', t37-t35)
            print('Time other', t4-t37)
            #values_extra=values_extra
            self.values[:,0]=self.values[:,0]*values_extra
            #self.values[:,0]=self.values[:,0]/np.sum(self.values[:,0])
        t5 = time.time()
        if plot_process:
            #assume b = 3
            ax = axn[1][1]
            figure, tax = ternary.figure(ax = ax, scale = 100)
            tax=self.plot_theta_score(tax,compute=False)
        self.lines=self.lines[:-batch_size]
        self.end_points=self.lines[:-batch_size]
        self.values=values
        for i in range(batch_size):
            self.create_line_from_point_nd(dim,len(self.points)-batch_size+i,theta_range,theta_distribution)
        values_extra=self.prod_theta_score_n(dim=dim,increment=increment,k=np.pi,return_type='to_include',a=self.points[-batch_size:],b=self.end_points[-batch_size:])
        self.values[:,0]=self.values[:,0]*values_extra
        if plot_process:
            figure, tax = ternary.figure(ax = axn[0][0], scale = 100)
            self.set_labels(method='1_batch',batch_size=batch_size)
            tax = self.plot(True,True,True,show=False,compute_theta=False,tax=tax)
            figure, tax = ternary.figure(ax = axn[1][2], scale = 100)
            tax=self.plot_theta_score(tax,compute=False,points=[0])
            plt.savefig('test.eps',dpi=800)
            plt.clf()
        if plot:
            self.set_labels(method='1_batch',batch_size=batch_size)
            self.plot(True,True,True,angle_product_heatmap=True,
                      title = 'heatmap', show=False, save=True,
                      name = plot,compute_theta=False)
        t6 = time.time()
        print('Time prod', t1-t0)
        print('Time loop', t5-t1)
        print('Time fin', t6-t5)
        return self.distance_to_optimal_f()

    def f_score(self,f,frac=0.9,method='sorted_score_at_frac'):
        sorted_f=np.sort(f)
        if method == 'sorted_score_at_frac':
            index=int(round(len(f)*frac))
            result = sorted_f[index]
        if method == 'skewness':
            result = scipy.stats.skew(sorted_f)
        return result

    def f_score_test(self,n=10000,dim=3,num_points=50,theta_range=30,
                 theta_distribution='uniform',increment=1):
        #generate n random setups, sort them by f and save heatmaps
        name_s = 'f_score_test/theta_range'
        file = open(name_s + 'results.txt','w+')
        scorebmeans = []
        distancemeans = []
        scorebstd = []
        distancestd = []
        for theta in range(5,41,5):
            scoresb = []
            distances = []
            for i in range(n):
                name = name_s + '_' + str(theta) + '_' + str(i)
                self.add_random_initial(dim)
                self.add_random_goal(dim)
                self.lines = np.empty((0,dim))
                self.end_points = np.empty((0,dim))
                for i in range(num_points-1):
                    self.points=np.append(self.points,np.array([self.random_point(dim)]),axis=0)
                for i in range(num_points):
                    self.create_line_from_point_nd(dim,i,theta,theta_distribution)
                self.prod_theta_score_n(dim=dim,increment=increment,k=np.pi,return_type='with_index',create_grid=True)
                scoresb.append(self.f_score(self.values[:,0],method='skewness'))
                distances.append(self.distance_to_optimal_f())
                self.set_labels(method='random')
                #self.plot(True,True,True,angle_product_heatmap=True,
                #          title = 'heatmap', show=False, save=True,
                #          name = name + '.eps',compute_theta=False)
                #plt.clf()
            scorebmeans.append(np.mean(scoresb))
            distancemeans.append(np.mean(distances))
            scorebstd.append(np.std(scoresb))
            distancestd.append(np.std(distances))
        fig,ax=plt.subplots()
        ax.errorbar(range(5,41,5),scorebmeans,scorebstd,ls='',
                marker='x',label='Skew based metric',capsize=5)
        ax.errorbar(range(5,41,5),distancemeans,distancestd,ls='',
                    marker='x',label='Distance to unknown from max(F)',capsize=5)
        plt.xlabel('Theta range')
        plt.legend()
        plt.show()

        #file.write('Theta: ' + str(theta) + ', Skew: ' + str(scoreb) + ', sortScore: ' + str(scoreb)
        #               + ', Distance: ' + str(distance) +'\n')
        file.close()



    def distance_to_optimal_f(self):
        sorted_values = self.values[self.values[:,0].argsort()]
        index = sorted_values[-1][1]
        f_max = self.omega[int(sorted_values[-1][1])]
        a = self.goal - f_max
        return np.linalg.norm(a)

    def plot_theta_score(self, tax, show=False ,compute=True,
                         points=None,point_labels=None,point_colors=None):
        if compute:
            self.prod_theta_score_n(return_type='with_index')
        f=self.values[:,0]
        f = f/np.sum(f)
        data = dict(zip(self.plotting_indicis,f))
        ta=time.time()
        if points is None:
            print('no points')
        else:
            if point_labels is None:
                sorted_values = self.values[self.values[:,0].argsort()]
                points=np.array([[self.goal],[self.omega[int(sorted_values[-1][1])]]])
                for i in points:
                    print(i)
                point_labels=['Target point','max(F)']
            for i in range(len(point_labels)):
                tax.scatter(points[i],
                        marker='x',s=50,linewidth = 1.,
                            zorder=2+i,label=point_labels[i])
            tax.legend()
        tax.heatmap(data,100)
        tb=time.time()
        print('plot time: ',tb-ta)
        tax.clear_matplotlib_ticks()
        tax.boundary(linewidth=1)
        tax.gridlines(color="black", multiple=10)
        tax.ticks(axis='lbr', linewidth=1, multiple = 10, fontsize=5)
        if show:
            tax.show()
        return tax

    def find_orthonormal(self,A):
        rand_vec=np.random.rand(A.shape[0],1)
        A = np.hstack((A,rand_vec))
        b = np.zeros(A.shape[1])
        b[-1] = 1
        x = np.linalg.lstsq(A.T,b,rcond=None)[0]
        return x/np.linalg.norm(x)

    def get_random_vec(self,x,length):
        #returns a random vector of length l constrained to the plane spanned
        #by x
        dim = x.shape[1]
        v_rand=[]
        if dim == 4:
            c0 = round(np.random.uniform(0,1),15)
            c1 = round(np.random.uniform(0,1),15)
            v_rand = c0*x[0] + c1*x[1]
            v_rand = length*v_rand/np.linalg.norm(v_rand)
        elif dim==3:
            v_rand = length*x[0]/np.linalg.norm(x[0])
        else:
            print('Error: function doesnt handle dimension ', dim)
        return v_rand

    def random_angle(self,upper, method):
        if method == 'uniform':
            p = (np.pi/180)*round(np.random.uniform(-upper,upper),10)
            return(p)

    def create_false_line(self,point_index):
        line = self.goal - self.points[point_index]
        line = line/np.linalg.norm(line)
        self.lines = np.append(self.lines,np.array([line]),axis=0)
        self.end_points = np.append(self.end_points,np.array([self.points[point_index]+line]),axis=0)

    def create_line_from_point_nd(self,dim,point_index,theta_range,
                              theta_distribution):
        #creates a line from a point using a method scalable to n dimensions
        v_perp = np.full((dim),1)
        true_line = self.goal - self.points[point_index]
        true_line_norm = true_line/np.linalg.norm(true_line)
        x=np.empty((dim-2,dim))
        A = np.stack((v_perp,true_line_norm),axis=1)
        for i in range(dim-2):
            x[i] = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x[i]]).T))
        theta = self.random_angle(theta_range,theta_distribution)
        v_rand = self.get_random_vec(x,np.tan(theta))
        estimated_line = (true_line_norm+v_rand)/np.linalg.norm(true_line_norm+v_rand)
        self.lines = np.append(self.lines,np.array([estimated_line]),axis=0)
        self.end_points = np.append(self.end_points,np.array([self.points[point_index]+estimated_line]),axis=0)

    def create_n_lines_from_domain(self,dim,domain,theta_range,theta_distribution):
        #Todo what?
        c = 2
        return None

    def random_point(self, dim, scale=100):
        point=[]
        if dim == 3:
            x = round(np.random.uniform(1, 99),10)
            y = round(np.random.uniform(1, 99-x),10)
            z = float(100 - x - y)
            point = [x,y,z]
        if dim == 4:
            x1 = round(np.random.uniform(0,100),10)
            x2 = round(np.random.uniform(0,100-x1),10)
            x3 = round(np.random.uniform(0,100-x1-x2),10)
            x4 = float(100 - x1 - x2 -x3)
            point = [x1,x2,x3,x4]
        random.shuffle(point)
        return np.array(point)

    def add_random_initial(self,dim):
        self.points = np.array([self.random_point(dim)])

    def add_random_goal(self,dim):
        self.goal = self.random_point(dim)

'''
@jit(nopython=True)
def block2(a,b,omega,m,count):
        a_broad = np.broadcast_to(a,(count,m,3))
        a_broad_t = np.swapaxes(a_broad,0,1)
        omega_broad = np.broadcast_to(omega,(m,count,3))
        a_omega = omega_broad-a_broad_t
        a_omega_normed = a_omega/np.moveaxis(np.broadcast_to(np.linalg.norm(a_omega,axis=2),(3,m,count)),0,2)
        ab=np.subtract(b,a)
        ab_norm = np.linalg.norm(ab,axis=1)
        ab_norm_t = np.swapaxes(np.broadcast_to(ab_norm,(3,m)),0,1)
        ab_normed = ab/ab_norm_t
        ab_normed_broad = np.swapaxes(np.broadcast_to(ab_normed,(count,m,3)),0,1)
        theta = np.arccos((ab_normed_broad*a_omega_normed).sum(axis=2))
        theta_p = np.exp(-1*theta/k)
        values = np.prod(theta_p,axis=0)
        #block 3
        '''


