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
from plotter import *
from errorpropagator import *
from visualisesquare import *
from wtconversion import *
from scipy import interpolate
SQRT3OVER2 = np.sqrt(3) / 2.

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

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

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

    def setup(self,normal_vectors,contained_point,cube_size,sigma,
              create_heatmap=False):
        self.sigma=sigma
        self.create_omega_constrained(normal_vectors,cube_size,contained_point,
                                     create_heatmap=create_heatmap)

    def getdistance(self):
        return np.sqrt(np.sum((self.goal-self.optimal_point)**2))

    def add_point(self,point):
        self.points = np.vstack([self.points,point])

    def add_random_point(self,dim):
        npoint = np.array([self.random_point(dim)])
        self.points = np.append(self.points,npoint,axis=0)

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
        if method == 'custom_figure':
            self.labels = {'Initial samples':[0,101]}
        if method == 'many lines one point':
            self.labels = {'point':[0,len(self.points)]}


    def add_average_comp(self):
        avg = np.array([0,0,0])
        for i in self.points:
            avg = i+avg
        avg = avg/3
        self.points = np.append(self.points,np.array([avg]),axis=0)

    def add_point_in_ball(self):
        x = self.points[-1]
        y = x + [-6,4,2]
        self.points = np.append(self.points,np.array([y]),axis=0)

    def add_line_from_ball_through_sample(self,form):
        if form == 'fake':
            v = self.points[0] - self.points[-1]
        if form == 'real':
             v = self.points[0] - self.points[-2]
             self.points=np.append(self.points,np.array([self.points[0]+0.3*v]),axis=0)
             print('ooo')
             print(self.points)
        return v

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
                        min_steps = -1*pi/vi
            final_point = np.add(point,line*min_steps)
            ab = final_point-point
            printid = True
            for i in ab:
                if i != 0:
                    printid = False
            if printid:
                print('here',ab)
                print(line)
                print(min_steps,'m')
            self.end_points = np.append(self.end_points,[final_point],axis=0)

    def plot(self,goal,points,lines,
             optimal_point = False,angle_product_heatmap = False,
             title = '', show=True, save=False, scale=100,
             name = 'test.png',compute_theta=True ,tax=None, ball=False,
             line=False):
        if tax is None:
            figure, tax = ternary.figure(scale=scale)
        tax.set_title(title,fontsize = 10)
        number_colours = 0
        subplots = False
        if (lines):
           # number_colours += len(self.lines)
            number_colours += 1
        if goal:
            number_colours += 1
        if  points:
            if self.labels:
                number_colours+=len(self.labels)+1
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
        #firgure gen
        if line:
            self.plot_line(tax,cmap,used_colours,self.points[-1],
                           self.add_line_from_ball_through_sample('fake'))
            self.plot_line(tax,cmap,used_colours-1,self.points[-2],
                           self.add_line_from_ball_through_sample('real'))
        if ball:
            self.plot_points(tax,cmap,used_colours,ball=True)
        #end
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
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
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
            '''
            tax.line(self.points[n], c_edge_intersect[n], linewidth=0.5,
                     color=cmap(n+used_colours), linestyle="-",label=str(n))
            '''
            tax.line(self.points[n], c_edge_intersect[n], linewidth=0.5,
                     color=cmap(used_colours+1), linestyle="-")
        return tax

    def plot_line(self,tax,cmap,used_colours,point,line):
        c_edge_intersect = np.empty((0,self.points.shape[1]))
        min_steps=9999
        for pi,vi in zip(point,line):
            if vi < 0:
                if pi/(-1*vi) < min_steps:
                    min_steps = -1*pi/vi
        final_point = np.add(point,line*min_steps)
        c_edge_intersect = np.append(c_edge_intersect,[final_point],axis=0)
        tax.line(point, c_edge_intersect[0], linewidth=0.5,
                 color=cmap(2+used_colours), linestyle="-")

        #tax.line(point,point+line)
        return tax



    def plot_points(self,tax,cmap,used_colours,ball=False):
        if not ball:
            print(self.labels)
            print('poo')
            for n,key in enumerate(self.labels):
                print(n)
                tax.scatter(self.points[self.labels[key][0]:self.labels[key][1]],
                            marker='x',s=30,linewidth = 0.5,
                            color=cmap(used_colours+n),
                            label = key,zorder=5)
            return tax
        else:
            print('here')
            tax.scatter([self.points[-2]],marker='o',s=500,zorder=0)
            return tax

    def plot_optimal_point(self,tax,colour):
        tax.scatter([self.optimal_point], marker='x', s=50,
                    color=colour,label='optimal')
        return tax

    def plot_goal(self,tax,colour):
        tax.scatter([self.goal], marker='x', s=30, linewidth=0.5, color=colour,label='target')
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




    def reduce_omega(self,omega,point_index_start,point_index_end,backstep,cone_angle):
        print('start length: ', len(omega))
        cone_angle=np.pi*cone_angle/180
        a=self.points[point_index_start:point_index_end]
        b=self.end_points[point_index_start:point_index_end]
        ab=b-a
        a = a-backstep*ab
        l = len(omega)
        n = len(a)
        o = a.shape[1]
        print(a.shape[0])
        a_broad = np.broadcast_to(a,(l,n,o))
        omega_broad = np.swapaxes(np.broadcast_to(omega,(n,l,o)),0,1)
        ax = omega_broad-a_broad
        ax_norms=np.linalg.norm(ax,axis=2)
        ab_norms=np.broadcast_to(np.linalg.norm(ab,axis=1),(l,n))
        norms=ax_norms*ab_norms
        ab_broad=np.broadcast_to(ab,(l,n,o))
        dot=np.ones((l,n))
        np.divide((ab_broad*ax).sum(axis=2),norms,out=dot,where=norms!=0)
        dot=np.clip(dot,-1,1)
        theta = np.arccos(dot)
        rows,cols = np.where(theta>cone_angle)
        rows=np.unique(rows)
        print(len(omega)-len(rows))
        omega=np.delete(omega,rows,axis=0)
        print('end length: ', len(omega))
        return omega




    def prod_theta_score_n(self,k=np.pi,dim=4,increment=10,
                           return_type = None, timed=False,
                          a=None,b=None,reduce=False):
        #Optimise so that End points MUST be 1 away from start points
        #block 1
        if a is None:
            self.omega = self.create_omega(increment,dim)
            print('nnnn')
            a = self.points
            b = self.end_points
        if reduce:
            self.omega = self.reduce_omega(self.omega,0,len(a),0.5,45)
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
        np.divide((ab_normed_broad*a_omega).sum(axis=2),
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

    def get_points_targets_for_exploration_evaluation(
        self,method='inc_exploration',a = 0.2,num_points=10,num_targets=100):
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
        #f = self.values[:,0]/np.sum(self.values[:,0])
        f=self.values/np.sum(self.values)
        f_broad = np.broadcast_to(f,(m,n,l))
        f_broad_masked = ma.masked_array(f_broad, mask = masked_angle.mask)
        weights = f_broad_masked.sum(axis=2)
        weights=weights/np.sum(weights,axis=0)
        return weights

    def expected_gain_in_info(self,k=np.pi):
        #f = self.values[:,0]
        f=self.values
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
        #original_score = scipy.stats.skew(np.sort(f))
        original_score = self.f_score(f,method='variance')
        post_f = f_broad*info
        post_f=post_f/np.moveaxis(np.broadcast_to(np.sum(post_f,axis=2),(l,m,n)),0,2)
        #post_score=scipy.stats.skew(np.sort(post_f,axis=2),axis=2)
        post_score=np.empty((post_f.shape[0],post_f.shape[1]))
        for n,i in enumerate(post_f):
            for m,j in enumerate(i):
                post_score[n][m]=self.f_score(j,method='variance')
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
        self.create_line_from_point_nd(dim,0,theta_range,theta_distribution,charge_constraint=charge_constraint)
        target_points = self.sampleonline(batch_size,0)
        for i in range(target_points[0],target_points[1]):
            self.create_line_from_point_nd(dim,i,theta_range,theta_distribution,charge_constraint=charge_constraint)


    def evaluation(self,dim,num_points,num_targets,theta_range,theta_distribution, angular_equivalence,increment,considered_fraction,batch_size,k,plot='',plot_process=False):
        if plot_process:
            if dim != 3:
                print('Error,np.std cant plot for dim != 3')
                return
        self.usual_setup(dim,theta_range,theta_distribution,batch_size)
        f, axn = (None,None)
        if plot_process:
            f, axn = plt.subplots(2,3)
            f.set_figheight(15)
            f.set_figwidth(30)
        t0 = time.time()
        self.prod_theta_score_n(dim=dim,increment=increment,k=np.pi,return_type='with_index',reduce=False)
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
            plt.savefig('../data/evaluation_opt/3d/process/test.eps',dpi=800)
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

    def f_score(self,fs,frac=0.9,method='sorted_score_at_frac'):
        if method == 'sorted_score_at_frac':
            sorted_f=np.sort(fs)
            index=int(round(len(fs)*frac))
            result = sorted_f[index]
        if method == 'skewness':
            sorted_f=np.sort(fs)
            result = scipy.stats.skew(sorted_f)
        if method == 'variance':
            mean=self.get_mean(fs)
            result=np.sum(fs*np.linalg.norm(mean-self.omega))
        if method=='max_variance':
            centre=self.get_max(fs)
            result=np.sum(fs*np.linalg.norm(centre-self.omega))

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
            self.prod_theta_score_n(dim=3,return_type='with_index',increment=1)
        f=self.values[:,0]
        f = f/np.sum(f)
        data = dict(zip(self.plotting_indicis,f))
        ta=time.time()
        if points is None:
            print('no points')
        else:
            if point_labels is None:
                sorted_values = self.values[self.values[:,0].argsort()]
                points=np.array([[self.goal],[self.omega[int(sorted_values[-1][1])]],[self.get_mean()]])
                for i in points:
                    print(i)
                point_labels=['Target point','max(F)','mean(F)']
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
        dof = x.shape[0]
        v_rand=[]
        if dof == 2:
            c0 = round(np.random.uniform(-1,1),15)
            c1 = round(np.random.uniform(-1,1),15)
            v_rand = c0*x[0] + c1*x[1]
            v_rand = length*v_rand/np.linalg.norm(v_rand)
        elif dof==1:
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
        self.end_points = np.append(self.end_points,
                                    np.array([self.points[point_index]+line]),axis=0)
        return

    def create_line_from_point_nd(self,dim,point_index,theta_range,
                              theta_distribution):
        v_perp = np.full((dim),1)
        true_line = self.goal - self.points[point_index]
        true_line_norm = true_line/np.linalg.norm(true_line)
        #creates a line from a point using a method scalable to n dimensions
        x=np.empty((dim-2,dim))
        A = np.stack((v_perp,true_line_norm),axis=1)
        for i in range(dim-2):
            x[i] = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x[i]]).T))
        print(A)
        theta = self.random_angle(theta_range,theta_distribution)
        v_rand = self.get_random_vec(x,np.tan(theta))
        estimated_line = (true_line_norm+v_rand)/np.linalg.norm(true_line_norm+v_rand)
        self.lines = np.append(self.lines,np.array([estimated_line]),axis=0)
        self.end_points = np.append(self.end_points,np.array([self.points[point_index]+estimated_line]),axis=0)

    def create_line_from_sample(self,dim,point_index):
        v_perp = np.full((dim),1)
        true_line = self.goal - self.points[point_index]
        true_line_norm = true_line/np.linalg.norm(true_line)
        #creates a line from a point using a method scalable to n dimensions
        x=np.empty((dim-2,dim))
        A = np.stack((v_perp,true_line_norm),axis=1)
        for i in range(dim-2):
            x[i] = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x[i]]).T))
        ep=error_propagator(dim)
        ep.initialise('b',self.goal,self.points[0])
        vs = np.empty((x.shape[0],x.shape[1]))
        for n,i in enumerate(x):
            std=ep.get_std(i)
            scale=np.random.normal(loc=0,scale=std)
            vs=np.append(vs,np.array([scale*x[n]]),axis=0)
        estimated_line=true_line_norm
        for i in vs:
            estimated_line=estimated_line+i
        self.lines = np.append(self.lines,np.array([estimated_line]),axis=0)
        self.end_points=np.append(self.end_points,np.array([self.points[point_index]+100*estimated_line]),axis=0)



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
        if dim == 5:
            x1 = round(np.random.uniform(0,100),10)
            x2 = round(np.random.uniform(0,100-x1),10)
            x3 = round(np.random.uniform(0,100-x1-x2),10)
            x4 = round(np.random.uniform(0,100-x1-x2-x3),10)
            x5 = float(100-x1-x2-x3-x4)
            point = [x1,x2,x3,x4,x5]
        random.shuffle(point)
        return np.array(point)

    def add_random_initial(self,dim):
        self.points = np.array([self.random_point(dim)])

    def add_initial(self,dim=3,p=[[30,20,50]]):
        self.points = p

    def add_random_goal(self,dim):
        self.goal = self.random_point(dim)

    def add_goal(self,goal,dim=3):
        self.goal = goal

    def get_mean(self,f=None,omega=None):
        if omega is None:
            omega=self.omega
        if f is None:
            f=self.values[:,0]
        u=np.swapaxes(np.broadcast_to(f,(omega.shape[1],len(f))),0,1)*omega
        mean=np.sum(u,axis=0)
        return mean

    def get_max(self,f=None,omega=None):
        if omega is None:
            omega=self.omega
        if len(f) != len(omega):
            print('Error - lengh of values must equal length of omega')
        return omega[np.argmax(f)]

    def set_charge_constraint_normal(self, normal):
        self.charge_normal = normal

    def charge_constraint_test(self,dim,normal_vectors,cube_size,contained_point,theta_range,theta_distribution,batch_size):
        self.usual_setup_constrained(dim,normal_vectors,cube_size,contained_point,theta_range,theta_distribution,batch_size)

    def create_omega_constrained(self, normal_vectors, cube_size,
                                 contained_point, create_heatmap=False):
        dim=normal_vectors.shape[1]
        plane_dim = dim-len(normal_vectors)
        max_length = np.sqrt(dim*cube_size**2)
        max_co = math.floor(max_length)
        normal_a = normal_vectors[0]
        normal_b = normal_vectors[1]
        x=np.empty((plane_dim,dim))
        A = np.stack((normal_a,normal_b),axis=1)
        for i in range(plane_dim):
            x[i] = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x[i]]).T))
        ii = np.array(range(-max_co,max_co+1,1))
        if plane_dim == 3:
            all_combinations=cartesian((ii,ii,ii))
        elif plane_dim ==2:
            all_combinations=cartesian((ii,ii))
        all_combinations_s=np.einsum('...i,ij->...j',all_combinations,x)
        all_combinations_s=all_combinations_s+contained_point
        omega=np.delete(all_combinations,np.where(np.any(
            (all_combinations_s>cube_size)|(all_combinations_s<0),
            axis=1)),axis=0)
        if plane_dim==2 and create_heatmap:
            jmin=np.amin(omega[:,1])
            jmax=np.amax(omega[:,1])
            imin=np.amin(omega[:,0])
            imax=np.amax(omega[:,0])
            self.xlim=[imin,imax]
            self.ylim=[jmin,jmax]
        #print("Number of grid points = ",omega.shape)
        self.omega=omega
        self.basis=x
        self.max_co=max_co
        self.constrained_dim=plane_dim
        self.contained_point=contained_point
        self.normal_vectors=normal_vectors
        self.cube_size=cube_size

    def random_point_constrained(self,n=1):
        choice = random.randrange(len(self.omega))
        indicis = range(len(self.omega))
        r_indicis = np.random.choice(indicis,n,replace=False)
        return self.omega[r_indicis]

    def add_random_initial_constrained(self):
        self.points=self.random_point_constrained()

    def add_random_goal_constrained(self):
        self.goal=self.random_point_constrained()[0]

    def usual_setup_constrained(self,dim,normal_vectors,cube_size,contained_point,theta_range,theta_distribution,batch_size):
        self.create_omega_constrained(normal_vectors,cube_size,contained_point)
        dim_constrained = dim - len(normal_vectors)
        self.add_random_initial_constrained()
        for i in range(batch_size-1):
            self.points=np.append(self.points,self.random_point_constrained(),axis=0)
        self.add_random_goal_constrained()
        self.lines = np.empty((0,dim_constrained))
        self.end_points = np.empty((0,dim_constrained))
        for i in range(batch_size):
            self.create_line_from_point_nd_constrained(dim_constrained,i,theta_range,theta_distribution)
        self.prod_theta_score_n_constrained(return_type='with_index')

    def create_line_from_point_nd_constrained(self,dim,point_index,theta_range,
                              theta_distribution):
        true_line = self.goal - self.points[point_index]
        true_line_norm = true_line/np.linalg.norm(true_line)
        x=np.empty((dim-1,dim))
        A = np.transpose(np.array([true_line_norm]))
        for i in range(dim-1):
            x[i] = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x[i]]).T))
        theta = self.random_angle(theta_range,theta_distribution)
        v_rand = self.get_random_vec(x,np.tan(theta))
        estimated_line = (true_line_norm+v_rand)/np.linalg.norm(true_line_norm+v_rand)
        estimated_line_star=np.matmul(np.transpose(self.basis),estimated_line)
        print('0 check',np.sum(estimated_line_star),np.sum(estimated_line_star*[1,2,-2,-1,-1]))
        self.lines = np.append(self.lines,np.array([estimated_line]),axis=0)
        self.end_points = np.append(self.end_points,np.array([self.points[point_index]+estimated_line]),axis=0)

    def create_line_from_sample_constrained(self,dim,point_index,
                                        method='default',sigma=None,length=1):
        #function to generate a normalised random line for the sample specified by point
        #index. line points towards goal with wobble specified by sigma
        #appends line to self.lines and point+line to self.end_points
        true_line = self.goal - self.points[point_index]
        true_line_norm = true_line/np.linalg.norm(true_line)
        x=np.empty((dim-1,dim))
        A = np.transpose(np.array([true_line_norm]))
        for i in range(dim-1):
            x[i] = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x[i]]).T))
        if method == 'default':
            sigma=self.simulate_sample(method=method)
        stds = np.empty((len(x)))
        for i in range(len(x)):
            stds[i] = np.einsum('k,kl,l',x[i],sigma,x[i])
        perp_vec=np.zeros((dim))
        for i in range(len(x)):
            c = np.random.normal(loc=0,scale=stds[i])
            perp_vec=perp_vec+c*x[i]
        estimated_line=perp_vec+true_line_norm
        estimated_line_norm=estimated_line/np.linalg.norm(estimated_line)
        self.lines = np.append(self.lines,np.array([estimated_line]),axis=0)
        self.end_points=np.append(self.end_points,np.array([self.points[point_index]+length*estimated_line]),axis=0)

    def create_plot_lines(self):
        self.plot2d_lines=np.empty((len(self.lines),2,2))
        self.end_points=self.get_end_points('constrained')
        if len(self.end_points)==len(self.points):
            for i in range(len(self.lines)):
                self.plot2d_lines[i]=[self.points[i],self.end_points[i]]


        '''

        ep.initialise('a',goal_standard[0][0],sample_standard[0][0])
        x_standard=self.convert_to_standard_basis(x)
        vs = np.empty((x.shape[0],x.shape[1]))
        for n,i in enumerate(x_standard):
            std=ep.get_std(i[0])
            scale=np.random.normal(loc=0,scale=std)
            vs=np.append(vs,np.array([scale*x[n]]),axis=0)
        estimated_line=true_line_norm
        for i in vs:
            estimated_line=estimated_line+i
        self.lines = np.append(self.lines,np.array([estimated_line]),axis=0)
        self.end_points = np.append(self.end_points,np.array([self.points[point_index]+estimated_line]),axis=0)
        #Todo add test to get average angle between true and estimated line and
        #check against what it should be
        '''

    def prod_theta_score_n_constrained(self,k=np.pi,dim=4,return_type=None,timed=False,a=None,b=None,reduce=False):
        #Optimise so that End points MUST be 1 away from start points
        #block 1
        dim = self.omega.shape[1]
        if a is None:
            a = self.points
            b = self.end_points
        #to do constrained reduce
        '''
        if reduce:
            self.omega = self.reduce_omega(self.omega,0,len(a),0.5,45)
        '''
        omega=self.omega
        mesh_size = len(omega)
        m = len(b) #number of lines
        #block 2
        t1 = time.time()
        #broadcast a to repeated mesh_size times in its middle direction
        #(start point has no dependance on x)
        a_broad = np.broadcast_to(a,(mesh_size,m,dim))
        a_broad_t = np.swapaxes(a_broad,0,1)
        #broadcast omega to be repeated m times on axis 0,
        # (need 1 'omega' for every line
        omega_broad = np.broadcast_to(omega,(m,mesh_size,dim))
        #get the distance between a and x for every x
        a_omega = omega_broad-a_broad_t
        a_omega_norm = np.linalg.norm(a_omega,axis=2)
        #normalise so that 2-norm of axis 2 is 1
        #a_omega_normed = a_omega/np.moveaxis(
        #    np.broadcast_to(np.linalg.norm(a_omega,axis=2),(dim,m,mesh_size)),0,2)
        #get the vector from a to b (already contatined in self.lines????
        ab=np.subtract(b,a)
        #normalise
        ab_norm = np.linalg.norm(ab,axis=1)
        ab_norm_t = np.swapaxes(np.broadcast_to(ab_norm,(dim,m)),0,1)
        ab_normed = ab/ab_norm_t
        #broadcast to m by mesh_size by dim
        ab_normed_broad = np.swapaxes(np.broadcast_to(ab_normed,(mesh_size,m,dim)),0,1)
        dot=np.ones((m,mesh_size))
        np.divide((ab_normed_broad*a_omega).sum(axis=2),
                        a_omega_norm,out=dot,where=a_omega_norm!=0)
        dot = np.clip(dot,-1,1)
        theta = np.arccos(dot)
        theta_p = np.exp(-1*theta/k)
        #get the mesh_size long vector representing value for each point in omega
        #by taking product of lines for each x in omega
        theta_p=theta_p/np.swapaxes(np.broadcast_to(np.sum(theta_p,axis=1),(mesh_size,m)),0,1)
        values = np.prod(theta_p,axis=0)
        #normalise
        #values=values/np.sum(values)
        if return_type == 'with_index':
            index=np.arange(0,len(values),step=1,dtype='int_')
            values_index = np.stack((values,index),axis=1)
            self.values = values_index
            print(self.values.shape,'hey')
        elif return_type == 'to_include':
            return values
        else:
            self.values = values
        #block 3
        t2 = time.time()
        if timed:
            print('Block 2: ', t2-t1)

    def convert_to_standard_basis(self,omega):
        A=self.basis
        p_standard=self.contained_point+np.einsum('ji,...j->...i',A,omega)
        return p_standard

    def plot_jon_test(self,points=None):
        label=['0,1,2,3,4,5,6,7,8,9']
        if points is None:
            plotter=Plotter(5)
            plotter.create_fig()
            plotter.add_points(self.omega_standard)
            plotter.plot_points()
 
    def simulate_pawley(self,b=5):
        x=self.random_point_constrained(n=6)
        self.goal=x[0]
        x=x[1:]
        ds=np.empty((len(x)))
        for n,i in enumerate(x):
            ds[n] = np.linalg.norm(self.goal-i)
        x_sorted=x[np.argsort(ds)]
        for i in x_sorted:
            print(i,np.linalg.norm(self.goal-i))
        self.points=x_sorted
        self.pawley_closer=[0,2]
        self.pawley_further=[2,5]
    
    def set_pawley_rank(self,pawley_rank=None):
        if pawley_rank is None:
            pawley_rank=[[[2,1,1,2,1],[2,1,1,3,0]],
                         [[2,1,1,1.5,1.5]],
                         [[2,1,1,1,2],[2,1,1,0,3]]]
        pawley_rank_normal=[0]*len(pawley_rank)
        for n,i in enumerate(pawley_rank):
            rankeq = [0]*len(i)
            for m,j in enumerate(i):
                point = np.array(j)
                if np.dot(self.normal_vectors[0],point)!=0:
                    print('Error, given points do not obey constraints')
                else:
                    point=point/np.sum(point)
                    point=np.einsum('ij,j',self.basis,point)
                    rankeq[m]=point
            pawley_rank_normal[n]=rankeq
        self.pawley_rank = pawley_rank_normal

    def set_pawley_rank_s(self,pawley_rank):
        pawley_rank_normal=[0]*len(pawley_rank)
        for n,i in enumerate(pawley_rank):
            rankeq = [0]*len(i)
            for m,j in enumerate(i):
                point = np.array(j)
                point=point/np.sum(point)
                rankeq[m]=point
            pawley_rank_normal[n]=rankeq
        self.pawley_rank = pawley_rank_normal

    def incorporate_pawley(self,kind='closer_further',plot=True):
        thrown=np.empty((0),dtype=int)
        omega=self.convert_to_standard_basis(self.omega)
        self.reduced_omega=omega
        d=self.reduced_omega.shape[1]
        if kind == 'closer_further':
            print('Error, function probably wrong')
            closer=self.points[self.pawley_closer[0]:self.pawley_closer[1]]
            further=self.points[self.pawley_further[0]:self.pawley_further[1]]
            for i in closer:
                for j in further:
                    a=np.broadcast_to(i,(len(self.reduced_omega),d))
                    ax=a-self.reduced_omega
                    da=np.linalg.norm(ax,axis=1)
                    b=np.broadcast_to(j,(len(self.reduced_omega),d))
                    bx=b-self.reduced_omega
                    db=np.linalg.norm(bx,axis=1)
                    diff = da-db
                    thrown=np.append(thrown,np.where(diff>0)[0])
        if kind == 'arbitrary_rank':
            if len(self.pawley_rank)<2:
                print('Error, need at least 2 pawley ranks')
            else:
                for rank,closer in enumerate(self.pawley_rank):
                    if rank+1<len(self.pawley_rank): 
                        for further in self.pawley_rank[rank+1:]:
                            for i in closer:
                                for j in further:
                                    a=np.broadcast_to(i,(len(self.reduced_omega),d))
                                    ax=a-self.reduced_omega
                                    da=np.linalg.norm(ax,axis=1)
                                    b=np.broadcast_to(j,(len(self.reduced_omega),d))
                                    bx=b-self.reduced_omega
                                    db=np.linalg.norm(bx,axis=1)
                                    diff = da-db
                                    thrown=np.append(thrown,np.where(diff>0)[0])
        thrown=np.unique(thrown)
        indicis=np.arange(0,len(self.omega),1)
        kept=np.where(np.isin(indicis,thrown,invert=True)==True)
        self.reduced_omega=omega[kept]
        self.thrown_omega=omega[thrown]
        print('kept points: ',self.reduced_omega.shape[0])
        if plot:
            self.plot_pawley_closer()

    def get_uniform_from_pawley(self,method='7',plot=True):
        reduced_omega_projected=np.empty((len(self.reduced_omega),3))
        print(self.basis.shape)
        print(self.reduced_omega.shape)
        reduced_omega=self.reduced_omega
        for n,i in enumerate(reduced_omega):
            reduced_omega_projected[n]=np.matmul(self.basis,i)
        points_standard=np.empty((10,5))
        if method == 'reduced_omega':
            points_standard=reduced_omega
        if method == 'broken':
            print(reduced_omega_projected.shape)
            imax=-999
            imin=999
            jmax=-999
            jmin=999
            kmax=-999
            kmin=999
            for point in reduced_omega_projected:
                if point[0] > imax:
                    imax=point[0]
                if point[0] < imin:
                    imin=point[0]
                if point[1] > jmax:
                    jmax=point[1]
                if point[1] < jmin:
                    jmin=point[1]
                if point[2] > kmax:
                    kmax=point[2]
                if point[2] < kmin:
                    kmin=point[2]
            print(imin,imax,jmin,jmax,kmin,kmax) 
            iplus=mean+np.array([(imax+imin)/4,0,0])
            iminus=mean-np.array([(imax+imin)/4,0,0])
            jplus=mean+np.array([0,(jmax+jmin)/4,0])
            jminus=mean-np.array([0,(jmax+jmin)/4,0])
            kplus=mean+np.array([0,0,(kmax+kmin)/4])
            kminus=mean-np.array([0,0,(kmax+kmin)/4])
            points=np.array([iplus,iminus,jplus,jminus,kplus,kminus,mean])
            print(points.shape,'nnn')
        if method == 'hmmm':
            imax=-999
            imin=999
            jmax=-999
            jmin=999
            kmax=-999
            kmin=999
            lmax=-999
            lmin=999
            mmax=-999
            mmin=999
            for n,point in enumerate(reduced_omega):
                if point[0] > imax:
                    imax=point[0]
                    points_standard[0]=point
                if point[0] < imin:
                    imin=point[0]
                    points_standard[1]=point
                if point[1] > jmax:
                    jmax=point[1]
                    points_standard[2]=point
                if point[1] < jmin:
                    jmin=point[1]
                    points_standard[3]=point
                if point[2] > kmax:
                    kmax=point[2]
                    points_standard[4]=point
                if point[2] < kmin:
                    kmin=point[2]
                    points_standard[5]=point
                if point[3] > lmax:
                    lmax=point[3]
                    points_standard[6]=point
                if point[3] < lmin:
                    lmin=point[3]
                    points_standard[7]=point
                if point[4] > mmax:
                    mmax=point[4]
                    points_standard[8]=point
                if point[4] < mmin:
                    mmin=point[4]
                    points_standard[9]=point
            #mean=np.array([(imax+imin)/2,(jmax+jmin)/2,(kmax+kmin)/2,
                          ##lmax+lmin)/2,(mmax+mmin)/2])
            mean=np.mean(reduced_omega,axis=0)
            points_standard[9]=mean
        if plot == True:
            print(points_standard.shape)
            plotter=Plotter(5)
            plotter.create_fig()
            plotter.set_projection(0,0)
            label=['0,1,2,3,4,5,6,7,8,9']
            plotter.add_points(points_standard)
            plotter.plot_points(c='green',s=1)
        return points_standard

    def plot_pawley_closer(self,thrown=False):
            plotter=Plotter(5)
            plotter.create_fig()
            plotter.set_projection(0,180)
            if thrown:
                plotter.add_points(self.thrown_omega)
                plotter.plot_points(show=False,c='red',s=1)
            plotter.add_points(self.reduced_omega)
            plotter.plot_points(c='green',s=1)

    def plot_points_jon(self,pawley_rank=False,points=None):
        plotter=Plotter(5)
        plotter.create_fig()
        if points is not None:
            cmap = get_cmap(len(points)+1)
            for label,point in enumerate(points):
                plotter.add_points([point])
                plotter.plot_points(color=cmap(label),marker='x',s=5,show=False,label=label)
        if pawley_rank:
            self.plot_pawley_ranking(plotter=plotter)


    def plot_pawley_test(self):
            closer=self.points[self.pawley_closer[0]:self.pawley_closer[1]]
            further=self.points[self.pawley_further[0]:self.pawley_further[1]]
            print('aa')
            print(closer)
            print(further)
            closer_standard=self.convert_to_standard_basis(closer)
            further_standard=self.convert_to_standard_basis(further)
            goal_standard=self.convert_to_standard_basis(self.goal)
            plotter=Plotter(5)
            plotter.create_fig()
            plotter.add_points(closer_standard)
            plotter.plot_points(show=False,c='green')
            plotter.add_points([goal_standard])
            plotter.plot_points(show=False,c='blue')
            plotter.add_points(further_standard)
            plotter.plot_points(c='red')
            self.incorporate_pawley()

    def plot_pawley_ranking(self,plotter=None):
        show=False
        if plotter is None:
            plotter=Plotter(5)
            plotter.create_fig()
        plotter.show_labels()
        cmap = get_cmap(len(self.pawley_rank)+1)
        for rank,i in enumerate(self.pawley_rank):
            if rank == len(self.pawley_rank)-1:
                show = True
            points=np.array(i)
            plotter.add_points(points)
            plotter.plot_points(show=show,label='Rank: ' +
                                str(rank),color=cmap(rank),s=1)

    def create_line_from_sample_test(self,normal_vectors,cube_size,
                                 contained_point,sigma=None):
        dim=4
        self.create_omega_constrained(normal_vectors,cube_size,contained_point)
        dim_constrained = dim - len(normal_vectors)
        self.add_random_initial_constrained()
        self.add_random_goal_constrained()
        self.lines = np.empty((0,dim_constrained))
        self.end_points = np.empty((0,dim_constrained))
        self.create_line_from_sample_constrained(dim_constrained,0,method='use_sigma',sigma=sigma)
        for i in range(500):
            self.add_point([self.points[0]])
            self.create_line_from_sample_constrained(dim_constrained,i+1,method='use_sigma',sigma=sigma)
        plotter=visualise_square()
        self.create_plot_lines()
        plotter.test_fig(self.plot2d_lines,self.points[0])


    def create_line_from_sample_test_uncon(self,dim=3):
        self.add_initial(dim=dim,p=np.array([[33,33,34]]))
        for i in range(1000):
            self.add_point(np.array([[33,33,34]]))
        self.add_goal(np.array([20,20,60]))
        self.lines = np.empty((0,dim))
        self.end_points = np.empty((0,dim))
        for i in range(101):
            self.create_line_from_sample(dim,i)
        self.set_labels(method='custom_figure')
        self.plot(True,True,True)
        N=50
        z=0.3
        u0=self.goal-self.points[0]
        u0=self.goal/np.linalg.norm(u0)
        mean=self.points[0]+z*u0
        sigma=np.array([[0.1,0,0],
                        [0,1,0],
                        [0,0,1]])
        X = np.linspace(-3, 3, N)
        Y = np.linspace(-3, 3, N)
        X, Y = np.meshgrid(X, Y)
        pos=np.empty(X.shape+(2,))
        pos[:,:,0]=X
        pos[:,:,1]=Y
        omega=self.create_omega(dim=3,increment=1)
        print(omega.shape)
        n = 3
        Sigma_det = np.linalg.det(sigma)
        Sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mean, Sigma_inv,
                        pos-mean)

        Z = np.exp(-fac / 2) / N
        #plot that fuzzy ball bro

    def plot_square_test(self):
        wt_convert = wt_converter()
        error_propagate = error_propagator(dim=4)
        plotter=visualise_square()
#        plotter.plot_omega(self.omega)
        #formula_a="Cs 1 Bi 1 Se 1 I 2"
        formula_a="Cs 1 Bi 1 Se 0 I 4"
        formula_b="Cs 2 Bi 1 Se 2 I 1"
        formula_c="Cs 1 Bi 2 Se 2 I 3"
        formulas=[formula_a,formula_b,formula_c]
        weights=[0.2,0.2,0.6]
        #get the moles and standard deviation from wt%
        moles,moles_error,formulas_standard=wt_convert.wt_to_moles(formulas,
                                                                   weights)
        error_propagate.set_moles_error(moles,formulas_standard,moles_error)
        merged_ball=error_propagate.get_merged_balls_p(self.basis)
        #plotter.plot_ball(merged_ball)

        small_balls=error_propagate.get_small_balls_p(self.basis)
        plotter.draw_ball_fig(merged_ball,small_balls,self.omega,self.cube_size)

    def gaussian_score(self):
        omega=self.omega
        self.values=np.array([1]*len(omega))

    def make_heatmap_constrained(self,values=None,omega=None):
        self.heatmap=np.zeros(
            (self.ylim[1]-self.ylim[0]+1,self.xlim[1]-self.xlim[0]+1))
        if values is None:
            values=self.values
        if omega is None:
            omega=self.omega
        for point,value in zip(omega,values):
            i=point[0]-self.xlim[0]
            j=point[1]-self.ylim[0]
            self.heatmap[j,i]=value
        return self.heatmap

    def heatmap_test(
        self,normal_vectors,cube_size,contained_point,sigma,method='p'):
        dim_constrained=normal_vectors.shape[1]-normal_vectors.shape[0]
        self.create_omega_constrained(normal_vectors,cube_size,
                                      contained_point,create_heatmap=True)
        self.add_random_initial_constrained()
        #self.add_point(np.array(self.random_point_constrained()))
        #self.add_point(np.array(self.random_point_constrained()))
        self.add_random_goal_constrained()
        self.lines = np.empty((0,dim_constrained))
        self.end_points = np.empty((0,dim_constrained))
        self.create_line_from_sample_constrained(dim_constrained,0,method='use_sigma',sigma=sigma,length=40)
        #self.create_line_from_sample_constrained(dim_constrained,1,method='use_sigma',sigma=sigma,
        #                                        length=40)
        #self.create_line_from_sample_constrained(dim_constrained,2,method='use_sigma',sigma=sigma,length=40)
        if method == 'p':
            basises=np.empty((self.lines.shape[0],self.lines.shape[1],self.lines.shape[1]))
            for i in range(len(self.lines)):
                basises[i]=self.get_basis(self.lines[i])
            self.set_sigma_simulated(sigma,1)
            stdses=self.get_stdses(basises)
            self.create_p(self.omega,basises,self.points,stdses,save_reduced=True)
            mean=self.get_mean(self.values_reduced,self.omega_reduced)
        elif method == 'angle':
            self.prod_theta_score_n_constrained(return_type='normal')
            mean=self.get_mean(self.values,self.omega)

        #self.gaussian_score()
        self.make_heatmap_constrained()
        plotter=visualise_square()
        #plotter.plot_heatmap(self.heatmap,self.xlim,self.ylim)
        self.create_plot_lines()
        #plotter.test_fig(self.plot2d_lines,self.points[0])
        #goal=self.convert_to_standard_basis(self.goal)
        #print(goal.shape)
        plotter.test_fig(self.goal,self.points,self.plot2d_lines,self.heatmap,self.xlim,self.ylim,self.omega,mean)

    def make_p_gaussian(self,sigma,scale,delta,save_reduced=False):
        self.sigma=sigma
        self.scale=scale
        self.delta=delta
        basises=np.empty((self.lines.shape[0],self.lines.shape[1],self.lines.shape[1]))
        for i in range(len(self.lines)):
            basises[i]=self.get_basis(self.lines[i])
        self.set_sigma_simulated(sigma,scale)
        stdses=self.get_stdses(basises)
        self.create_p(
            self.omega,basises,self.points,stdses,delta=delta,
            save_reduced=save_reduced)
        
    def get_stdses(self,basises):
        #function to compute the standard deviation in each of the directions
        #in self.basis (but not the the first one) and store as a n,dim-3
        stdses=np.empty((basises.shape[0],basises.shape[1]-1))
        for n,basis in enumerate(basises):
            for m,i in enumerate(basis[1:]):
                stdses[n][m]=np.einsum('i,ij,j',i,self.sigmas[n],i)
        return stdses


    def get_basis(self, estimated_direction):
        dim = len(estimated_direction)
        #function to caluclate basis for each sample
        #basis: numpy (dim,dim) array such such that
        # basis matmul (a point) gives the point in
        # representation where first element is size in estimated_direction and
        # subsequent elements are sizes in orthogonal directions
        x=np.empty((dim-1,dim))
        A = np.transpose(np.array([estimated_direction]))
        for i in range(dim-1):
            x[i] = self.find_orthonormal(A)
            A = np.hstack((A,np.array([x[i]]).T))
        return A.T

    def set_sigma_simulated(self,sigma,scale=1):
        self.sigmas=np.zeros((len(self.points),self.points.shape[1],self.points.shape[1]))+sigma/scale

    def create_p(self,omega,basises,samples,stdses,
                 delta=1,save_reduced=False,addition=False):
        #function to calculate P for every point in omega
        #omega: numpy (o,dim) array for every point
        #basises: numpy (n,dim,dim) array for n samples such that
        # the matrix (i,dim,dim) matmul (a point) gives the point in
        # representation where first element is size in perp direction and
        # subsequent elements are sizes in orthogonal directions
        #samples: numpy (n,dim) array giving location of each sample
        #stdss: numpy (n,dim-1) array giving std of each sample in its
        # orthogonal directions
        #Indexes:
            #o-omega
            #n-sample
            #dim dimensions
        o = len(omega)
        n = len(samples)
        m = stdses.shape[1]
        dim=self.points.shape[1]

        #reduce omega to only consider points which are in forward orthogonal
        #direction9
        omega_broad=np.swapaxes(np.broadcast_to(omega,(n,o,dim)),0,1)
        samples_broad=np.broadcast_to(samples,(o,n,dim))
        ax=omega_broad-samples_broad
        orth_direction_broad=np.broadcast_to(basises[:,0,:],(o,n,dim))
        ax_perp=np.sum(ax*orth_direction_broad,axis=2)
        selected_indexes=np.all(ax_perp >0,axis=1)
        ax_perp_reduced=ax_perp[selected_indexes]
        ax_reduced=ax[selected_indexes]
        o=len(ax_reduced)

        #project points
        ax_perp_broad=np.moveaxis(np.broadcast_to(ax_perp_reduced,(dim,o,n)),0,2)
        #ax_proj=np.zeros((o,n,dim))
        #np.divide(ax_reduced,ax_perp_broad,where=ax_perp_broad!=0,out=ax_proj)
        parallel_directions=basises[:,1:,:]
        ax_parallel=np.einsum('...id,ikd->...ik',ax_reduced,parallel_directions)

        #scale sigmas
        scaled_std=np.einsum('...i,k...->k...i',stdses,ax_perp_reduced)

        #lower = ax_parallel-delta
        #CDF_lower=scipy.stats.norm(loc=0,scale=scaled_std).cdf(lower)
        #upper = ax_parallel+delta
        #CDF_upper=scipy.stats.norm(loc=0,scale=scaled_std).cdf(upper)
        #p_reduced=CDF_upper-CDF_lower
        p_reduced=scipy.stats.norm.pdf(ax_parallel,loc=0,scale=scaled_std)
        for i in p_reduced[:,0,0]:
            if i <0:
                print(i)
        #normalisation?
        normalisation=np.sum(p_reduced,axis=0)
        p_reduced=p_reduced/np.broadcast_to(normalisation,p_reduced.shape)
        p_reduced=np.prod(p_reduced,axis=2)
        p_reduced=np.prod(p_reduced,axis=1)
        p=np.zeros((len(omega)))
        p[selected_indexes]=p_reduced
        if not addition:
            self.values=p
            if save_reduced:
                self.omega_reduced=self.omega[selected_indexes]
                self.values_reduced=p_reduced
        else:
            self.values=self.values*p



    def berny_test(
            self,normal_vectors,cube_size,contained_point,sigma,method='p',scale=1,delta=1):
        self.contained_point=contained_point
        self.normal_vectors=normal_vectors
        dim_constrained=normal_vectors.shape[1]-normal_vectors.shape[0]
        self.create_omega_constrained(normal_vectors,cube_size,
                                      contained_point,create_heatmap=True)
        self.add_random_initial_constrained()
        #self.add_point(np.array(self.random_point_constrained()))
        #self.add_point(np.array(self.random_point_constrained()))
        self.add_random_goal_constrained()
        self.lines = np.empty((0,dim_constrained))
        self.end_points = np.empty((0,dim_constrained))
        self.create_line_from_sample_constrained(dim_constrained,0,method='use_sigma',sigma=sigma,length=40)
        #self.create_line_from_sample_constrained(dim_constrained,1,method='use_sigma',sigma=sigma,length=40)
        #self.create_line_from_sample_constrained(dim_constrained,2,method='use_sigma',sigma=sigma,length=40)
        if method=='p':
            basises=np.empty((self.lines.shape[0],self.lines.shape[1],self.lines.shape[1]))
            for i in range(len(self.lines)):
                basises[i]=self.get_basis(self.lines[i])
            self.set_sigma_simulated(sigma)
            stdses=self.get_stdses(basises)
            self.create_p(self.omega,basises,self.points,stdses,save_reduced=True)
            #self.values=self.values/np.sum(self.values)
            data=self.convert_f_to_new_projection('berny',self.values_reduced,self.omega_reduced)
        if method=='angle':
            self.prod_theta_score_n_constrained(return_type='normal')
            data=self.convert_f_to_new_projection('berny',self.values,self.omega)
        #self.gaussian_score()
        #self.make_heatmap_constrained()
        plotter=Plotter('bernyterny')
        points=self.convert_points_to_new_projection('berny',self.points)
        end_points=self.get_end_points('berny')
        point_labels={'Initial':[0,1]}
        plotter.berny_testing(data,points,end_points,point_labels)

    def random_initialise(self,n):
        dim_constrained=(self.normal_vectors.shape[1]
                         -self.normal_vectors.shape[0])
        points=self.random_point_constrained(n=n+1)
        self.goal=points[0]
        self.points=points[1:]
        self.lines=np.empty((0,dim_constrained))
        self.end_points=np.empty((0,dim_constrained))
        for i in range(n):
            self.create_line_from_sample_constrained(
                dim_constrained,i,method='use_sigma',sigma=self.sigma)

    def convert_to_ternary_basis(self,x):
        x_t=np.empty((len(x),2))
        for n,i in enumerate(x):
            x_tt = [i[0]+i[1]/2,SQRT3OVER2*i[1]]
            x_t[n]=x_tt
        return x_t

    def get_estimated_known_composition(self,point):
        point_s=self.convert_to_standard_basis(point)
        goal_s=self.convert_to_standard_basis(self.goal)
        line = point_s-goal_s
        mindist=9999
        end_point=None
        for x,delta in zip(goal_s,line):
            if delta<0:
                distance=-x/delta
                if distance<mindist:
                    mindist=distance
        est_known=goal_s+mindist/2*line
        return est_known
        
    def make_formula_for_molar_mass(self,phases,composition):
        formula=""
        for p,c in zip(phases, composition):
            formula += p + " " + str(c) + " "
        return formula


    def get_end_points(self,method):
        if method == 'berny':
            points=self.convert_points_to_new_projection('berny',self.points)
            end_points=self.convert_points_to_new_projection('berny',self.end_points)
            lines=end_points-points
            constraint=self.normal_vectors[0][:3]
            end_points_t=np.empty((points.shape[0],points.shape[1]+1))
            for i in range(len(self.points)):
                start=np.append(points[i],100-np.sum(points[i]))
                delta=np.append(lines[i],-np.sum(lines[i]))
                distance_to_boundary=-start/delta
                mindist=99999
                for j in range(len(start)):
                    if delta[j] < 0:
                        if distance_to_boundary[j]<mindist:
                            mindist=distance_to_boundary[j]
                end_points_t[i]=start+mindist*delta
            return end_points_t
        if method == 'constrained':
            points=self.convert_to_standard_basis(self.points)
            end_points=self.convert_to_standard_basis(self.end_points)
            lines=end_points-points
            end_points_t=np.empty(
                (len(end_points),self.normal_vectors.shape[1]))
            for n,(point,line) in enumerate(zip(points,lines)):
                mindist=99999
                for x,delta in zip(point,line):
                    if delta < 0:
                        distance=-x/delta
                        if distance<mindist:
                            mindist=distance
                end_points_t[n]=point+mindist*line
            end_points_s=end_points_t-self.contained_point
            A=self.basis
            end_points_s=np.einsum('ij,...j->...i',A,end_points_s)
            return end_points_s

                

    def convert_to_berny_basis(self,x):
        if x.ndim==2:
            amount=x[:,:3].sum(axis=1)
            normalised=100*x/(np.broadcast_to(amount,(4,len(x))).T)
            return normalised[:,:2]
        elif x.ndim==1:
            amount=np.sum(x[:3])
            normalised=100*x/amount
            return normalised[:2]
        else:
            print('Error - lick ma balls')


    def convert_f_to_new_projection(
        self,projection,f_values,f_locations,normalise=True):
        if len(f_values)!=len(f_locations):
               print('Error number of values does not match umber of locations')
        if projection=='berny':
            f_locations_stan=self.convert_to_standard_basis(f_locations)
            f_locations_to_tern=self.convert_to_berny_basis(f_locations_stan)
            (gridpoints_tuple,gridpoints)=self.create_grid_points()
            f=interpolate.griddata(f_locations_to_tern,f_values,gridpoints,method='cubic',fill_value=0)
            if normalise:
                f=f/np.sum(f)
            f[f<0]=0
            data=dict(zip(gridpoints_tuple,f))
            return data

    def convert_points_to_new_projection(self,projection,points):
        if projection=='berny':
            points_stan=self.convert_to_standard_basis(points)
            points_to_tern=self.convert_to_berny_basis(points_stan)
            return points_to_tern

    def create_grid_points(self):
        n = 101
        total = int(n*(n+1)/2)
        gridpoints_tuple = [None]*total
        gridpoints_array=np.empty((total,2))
        count=0
        for i in range(0,101):
            for j in range(0,101-i):
                gridpoints_tuple[count]=(i,j)
                gridpoints_array[count]=[i,j]
                count += 1
        return (gridpoints_tuple,gridpoints_array)

    def get_score(self,method):
        if method=='d_g_mu':
            mean=self.get_mean(f=self.values)
            return np.linalg.norm(mean-self.goal)
        if method=='mean_mode_variance':
            values=self.values
            summ=np.sum(values)
            if summ!=0:
                mean=self.get_mean(f=values)
                mean_score=np.linalg.norm(mean-self.goal)
                mode=self.get_max(f=values)
                mode_score=np.linalg.norm(mode-self.goal)
                var=self.f_score(values,method='variance')
            else:
                print('Error p 0')
                mean_score=0
                mode_score=0
                var=0
            return np.array([mean_score,mode_score,var])


    def choose_next_best_point_a(self,power):
        #chooses next best as a sample from omega
        sample=self.values**power
        sample=sample/np.sum(sample)
        next_point=self.omega[np.random.choice(
            range(len(self.omega)),p=sample)]
        return next_point

    def choose_next_best_point_b(
            self,num_points,num_targets,angular_equivalence,ex_sigma,delta,
            num_chosen=1,exclusion=10,allow_reduce=True,
            use_reduced_omega=False):
        dim = self.constrained_dim
        values=self.values
        if use_reduced_omega:
            values=self.values_reduced
            self.values=self.values_reduced
            self.omega=self.omega_reduced
        if self.get_points_targets_for_exploration_evaluation_b(
                num_points=num_points,num_targets=num_targets,
                allow_reduce=allow_reduce):
            weights=self.weight_points_for_exploration_evaluation(
                cutoff = angular_equivalence)
            change_score = self.expected_gain_in_info_b(ex_sigma,delta)
            expected_score=np.sum(weights*change_score,axis=0)
            sorted_trial_points=self.exploration_points[expected_score.argsort()]
            if num_chosen == 1:
                return sorted_trial_points[-1]
            else:
                chosen_points=np.empty((0,dim))
                chosen_points=np.append(chosen_points,[sorted_trial_points[-1]],axis=0)
                choosing=True
                i=-2
                while(choosing):
                    outside=True
                    p=sorted_trial_points[i]
                    for n,point in enumerate(chosen_points):
                        if np.linalg.norm(p-point)<exclusion:
                            outside=False
                    if outside:
                        chosen_points=np.append(chosen_points,[p],axis=0)
                        if len(chosen_points)==num_chosen:
                            choosing=False
                    i=i-1
                return chosen_points
        else:
            return None

    def choose_next_best_points_sphere(
            self,centre,n,allowed_angle,radius_reduction=None,slope=None,
            intercept=None,set_radius=None):
        points=self.get_points_on_sphere(n-1,allowed_angle)
        f=self.values/np.sum(self.values)
        std=np.sqrt(self.f_score(f,method='variance'))
        radius=None
        if radius_reduction is not None:
            radius=std/radius_reduction
        elif slope is not None:
            radius=slope*std+intercept
        elif set_radius is not None:
            radius=set_radius
        else:
            print('Error: unknown radius method')
        #radius=1.5
        points=points*radius
        if centre == 'mean':
            centre=self.get_mean(f=f)
        elif centre == 'max':
            centre=self.get_max(f=f)
        else:
            print('Are you sure custom centre is valid?')
        points=points+centre
        points=np.append(points,[centre],axis=0)
        return points
        
    def eval_next_best_points_sphere_radius(
            self,centre,n,allowed_angle,radii_reduction=None,radii=None,
            custom_radius=None,slope=None,intercept=None):
        points=self.get_points_on_sphere(n,allowed_angle)
        f=self.values/np.sum(self.values)
        std=np.sqrt(self.f_score(f,method='variance'))
        radiuses=None
        if custom_radius is not None:
            radiuses=[]
            radiuses.append(radii)
            radiuses.append(std/radii_reduction)
            radiuses.append(slope*std+intercept)
            radiuses=np.array(radiuses)
        elif radii_reduction is not None:
            radiuses=(std/radii_reduction)
        elif radii is not None:
            radiuses=radii
        else:
            print('Error, no radius method')
        points=np.einsum('ij,...->...ij',points,radiuses)
        if centre == 'mean':
            centre=self.get_mean(f=f)
        elif centre == 'max':
            centre=self.get_max(f=f)
        else:
            print('Are you sure custom centre is valid?')
        points=points+centre
        return points
        
        '''
    def eval_best_radius(
            self,centre,n,allowed_angle,radii,num_trials=100):
        distances=np.empty((num_trials,0,0))
        for i in range(num_trials):
            points=self.get_points_on_sphere(n,allowed_angle)
            points=np.einsum('ij,...->...ij',points,radii)
        '''

    def get_points_on_sphere(self,n,allowed_angle):
        success = False
        attempts=0
        d=self.constrained_dim
        A=self.basis
        while not success:
            tchosen_points=np.empty((0,d))
            exit=False
            single_attempts=0
            while not exit:
                x=np.empty(d)
                for m in range(d):
                    x[m]=random.uniform(-1,1)
                x=x/np.linalg.norm(x)
                if len(tchosen_points)>0:
                    minangle=360
                    for y in tchosen_points:
                        angle=np.degrees(np.arccos(np.dot(x,y)))
                        if angle<minangle:
                            minangle=angle
                    if minangle>allowed_angle:
                        tchosen_points=np.append(
                            tchosen_points,[x],axis=0)
                        if len(tchosen_points)==n:
                            success=True
                            exit=True
                            chosen_points=tchosen_points
                else:
                    tchosen_points=np.append(tchosen_points,[x],axis=0)
                single_attempts=single_attempts+1
                if single_attempts>1000:
                    exit=True
            attempts=attempts+1
            if attempts>10000:
                print('Error cant find point')
                return False
        return chosen_points








    def get_points_targets_for_exploration_evaluation_b(
        self,num_points=10,num_targets=100,allow_reduce=True):
        p=self.values
        count_non_zero=np.count_nonzero(p!=0)
        if count_non_zero<num_points+num_targets:
            if allow_reduce:
                print('Minor error, num point,target has been reduced as',
                      ' not enough gridpoints')
                if count_non_zero>10:
                    num_points=int(np.floor(count_non_zero*0.1))
                    num_targets=int(np.floor(count_non_zero*0.9))
                else:
                    return False
            else:
                return False
        p=p/np.sum(p)
        points=self.omega[np.random.choice(
            range(len(self.omega)),p=p,size=num_points+num_targets,
            replace=False)]
        np.random.shuffle(points)
        self.exploration_points = points[:num_points]
        self.exploration_targets = points[num_points:]
        return True

    def expected_gain_in_info_b(self,ex_sigma,delta):
        #get basises
        n = len(self.exploration_points)
        m = len(self.exploration_targets)
        l = self.omega.shape[0]
        o = self.omega.shape[1]
        samples_broad=np.broadcast_to(self.exploration_points,(m,n,o))
        targets_broad=np.swapaxes(
            np.broadcast_to(self.exploration_targets,(n,m,o)),0,1)
        estimated_dirs=targets_broad-samples_broad
        basises=np.empty((m,n,o,o))
        for i,ii in enumerate(estimated_dirs):
            for j,jj in enumerate(ii):
                basises[i][j]=self.get_basis(jj)

        #get ax
        a_broad=np.broadcast_to(self.exploration_points,(l,n,o))
        x_broad=np.swapaxes(np.broadcast_to(self.omega,(n,l,o)),0,1)
        ax = x_broad-a_broad

        #get ax perp and parallel
        ax_prime=np.einsum('ijkl,mjl->ijmk',basises,ax)

        #scale sigmas
        scaled_std=ex_sigma*ax_prime[:,:,:,0]
        scaled_std=np.moveaxis(np.broadcast_to(scaled_std,(o-1,m,n,l)),0,3)
        calculate_indices=np.where(scaled_std>0)
        std_to_calc=scaled_std[calculate_indices]
    

        #get bounds 
        ax_parallel_to_calc=ax_prime[:,:,:,1:][calculate_indices]
        lower = ax_parallel_to_calc-delta
        upper = ax_parallel_to_calc+delta

        #compute cff's
        out = np.empty(lower.shape)
        CDF_lower=scipy.stats.norm(loc=0,scale=std_to_calc).cdf(lower)
        CDF_upper=scipy.stats.norm(loc=0,scale=std_to_calc).cdf(upper)

        #get p
        p_from_calc=CDF_upper-CDF_lower
        p=np.zeros(scaled_std.shape)
        p[calculate_indices]=p_from_calc

        #Normalisation?
        normalisation=np.moveaxis(
            np.broadcast_to(np.sum(p,axis=2),(l,m,n,o-1)),0,2)
        p=p/normalisation

        #product over orthogonal directions
        p=np.prod(p,axis=3)

        #Get original score
        f=self.values
        original_score = self.f_score(f,method='variance')

        #get final score
        f_broad=np.broadcast_to(f,(m,n,l))
        post_p = f_broad*p
        #post_p=post_p/np.moveaxis(
        #    np.broadcast_to(np.sum(post_p,axis=2),(l,m,n)),0,2)
        post_score=np.empty((post_p.shape[0],post_p.shape[1]))
        for n,i in enumerate(post_p):
            for m,j in enumerate(i):
                summ=np.sum(j)
                if summ!=0:
                    j=j/summ
                    post_score[n][m]=self.f_score(j,method='variance')
                else:
                    print('hey')
                    post_score[n][n]=0
        change_score=original_score-post_score
        return change_score

    def sample_next_point(
            self,method,sigma,power=None,num_points=0,num_targets=0,delta=0,
            angular_equivalence=0,increment=1,considered_fraction=0.2,scale=0):
        if method == 'a':
            point=self.choose_next_best_point_a(power)
        elif method == 'b':
            point=self.choose_next_best_point_b(
                num_points,num_targets,angular_equivalence,increment,
                considered_fraction,sigma[0,0],delta)
        else:
            print('Unknown method')
        if point is None:
            print('Error, got to close')
            return False
        if np.all(point == self.goal):
            return False
        self.add_point(point)
        self.create_line_from_sample_constrained(
            self.constrained_dim,-1,method='use_sigma',sigma=sigma)
        return True
        
    def update_values(self,point_indexes,sigma,scale,delta):
        basises=np.empty(
            (len(point_indexes),self.lines.shape[1],self.lines.shape[1]))
        points=np.empty((len(point_indexes),self.lines.shape[1]))
        for i,index in enumerate(point_indexes):
            basises[i]=self.get_basis(self.lines[index])
            points[i]=self.points[index]

        self.set_sigma_simulated(sigma,scale)
        stdses=self.get_stdses(basises)
        self.create_p(
            self.omega,basises,points,stdses,delta=delta,addition=True)

    def sample_points_test(
            self,number_samples,normal_vectors,contained_point,cube_size,sigma,
            scale,delta,power,plot_process=False):
        num_points=10
        num_targets=100
        angular_equivalence=10
        increment=1
        considered_fraction=0.2
        method='b'
        self.setup(
            normal_vectors,contained_point,cube_size,sigma,
            create_heatmap=plot_process)
        self.random_initialise(1)
        self.make_p_gaussian(sigma,scale,delta)
        score=np.empty((number_samples+1))
        score[0]=self.get_score('d_g_mu')
        if plot_process:
            plotter=visualise_square()
            plotter.process_fig()
            self.make_heatmap_constrained()
            self.create_plot_lines()
            plotter.plot_heatmap(
                self.heatmap,self.xlim[0],self.xlim[1],self.ylim[0],
                self.ylim[1],use_axs=[0,0],show=False)

        for i in range(number_samples):
            unsuc = self.sample_next_point(
                method,sigma,power,num_points,num_targets,angular_equivalence,
                increment,considered_fraction,delta)
            if unsuc:
                if plot_process:
                    use_axs=[0,0]
                    if i < 3:
                        use_axs=[0,i]
                    else:
                        use_axs[1,i-3]
                    plotter.plot_scatter(
                    None,np.array([self.points[-1]]),show=False,lim=False,
                    marker='x',c='green',label='next chosen point',
                    use_axs=use_axs)
                self.update_values([-1],sigma,scale,delta)
                if plot_process:
                    use_axs=[0,0]
                    if i < 2:
                        use_axs=[0,i+1]
                    elif i>=5:
                        use_axs[1,i-2]
                    self.plot_scatter(
                        None,np.array([self.points[-1]]),show=False,lim=False,
                        marker='x',c='green',label='next chosen point')
                score[i+1]=self.get_score('d_g_mu')
            else:
                score[i+1]=0
        return score

    def plot_small_balls(self,means,sigmas,projection='None'):
        values = np.zeros(self.omega.shape[0])
        for mean,sigma in zip(means,sigmas):
            values = values + scipy.stats.multivariate_normal.pdf(
                self.omega,mean=mean,cov=sigma)
        if projection == 'None':
            heatmap=self.heatmap
            for point,value in zip(self.omega,values):
                i=point[0]-self.xlim[0]
                j=point[1]-self.ylim[0]
                heatmap[j,i]=value
            plotter=visualise_square()
            plotter.plot_heatmap(
                heatmap,self.xlim[0],self.xlim[1],self.ylim[0],self.ylim[1])
        if projection == 'berny':
            plotter=Plotter('bernyterny')
            data=self.convert_f_to_new_projection('berny',values,self.omega)
            plotter.small_balls_heat_fig(data)


    def make_merged_ball_values(self,mean,sigma):
        values = scipy.stats.multivariate_normal.pdf(
            self.omega,mean=mean,cov=sigma)
        return values

    def plot_merged_ball(
            self,mean,sigma,small_means,point_labels,mean_label,
            projection='None'):
        values=self.make_merged_ball_values(mean,sigma)
        if projection == 'None':
            heatmap=self.heatmap
            for point,value in zip(self.omega,values):
                i=point[0]-self.xlim[0]
                j=point[1]-self.ylim[0]
                heatmap[j,i]=value
            plotter=visualise_square()
            plotter.plot_heatmap(
                heatmap,self.xlim[0],self.xlim[1],self.ylim[0],self.ylim[1])
        if projection == 'berny':

            small_means=self.convert_points_to_new_projection(
                'berny',small_means)
            mean=self.convert_points_to_new_projection(
                'berny',np.array([mean]))[0]
            plotter=Plotter('bernyterny')
            data=self.convert_f_to_new_projection('berny',values,self.omega)
            plotter.merged_balls(data,mean,small_means,point_labels,
                                mean_label)

    def convert_point_to_constrained(self,point):
        point_s=point-self.contained_point
        point=np.einsum('ij,j',self.basis,point_s)
        return point

    def add_first_sample(self,point_s,mean,osigma):
        point_s=point_s-self.contained_point
        point=np.einsum('ij,j',self.basis,point_s)
        distance=np.linalg.norm(point-mean)
        sigma=np.diag(osigma/distance)
        self.points=np.broadcast_to(point,(1,self.constrained_dim))
        self.lines=np.broadcast_to(point-mean,(1,self.constrained_dim))
        self.sigmas=np.broadcast_to(
            sigma,(1,self.constrained_dim,self.constrained_dim))
        self.end_points=self.points+self.lines
        return np.mean(osigma/distance)

    def add_line(self,line):
        self.lines = np.vstack([self.lines,line])

    def add_sigma(self,sigma):
        self.sigmas = np.vstack((self.sigmas,[sigma]))

    def calculate_p_from_samples(self,delta,scale):
        basises=np.empty((
            self.lines.shape[0],self.lines.shape[1],self.lines.shape[1]))
        for i in range(len(self.lines)):
            basises[i]=self.get_basis(self.lines[i])
        stdses=self.get_stdses(basises)
        stdses*=scale
        self.create_p(self.omega,basises,self.points,stdses,delta=delta)
        self.sigma=self.sigmas[0]
        self.delta=delta
        self.scale=scale

    def plot_p(self,projection='None'):
        self.make_heatmap_constrained()
        plotter=visualise_square()
        if projection == 'None':
            plotter.plot_heatmap(
                self.heatmap,self.xlim[0],self.xlim[1],self.ylim[0],
                self.ylim[1])

    def plot_merged_ball_p(self,mean,sigma,delta,projection='None'):
        ball_values=self.make_merged_ball_values(mean,sigma)
        #ball_values=ball_values/(np.sum(ball_values))
        values=self.values
        #values=values/np.sum(values)
        heatmap=self.heatmap
        for point,value,ball_value in zip(self.omega,values,ball_values):
            i=point[0]-self.xlim[0]
            j=point[1]-self.ylim[0]
            heatmap[j,i]=value+ball_value
        if projection == 'None':
            plotter=visualise_square()
            plotter.plot_heatmap(
                heatmap,self.xlim[0],self.xlim[1],self.ylim[0],self.ylim[1])

    def sample_next_batch(
            self,method,sigma,scale,delta,num_points=None,
            num_targets=None,angular_equivalence=None,num_chosen=None,
            exclusion=None,plot_process=False,rietveld_closest=False,
            n_points=None,batch_size=None,return_points=False):

        if plot_process:
            goal_t=self.convert_points_to_new_projection('berny',self.goal)
            values=self.values/np.sum(self.values)
            data_t=self.convert_f_to_new_projection('berny',values,self.omega)
            plotter=Plotter("bernyterny")
            plotter.explore_batch(goal_t,data_t)

        next_points=None
        if method == 'lastline':
            next_points=np.empty((batch_size,self.constrained_dim))
            point=self.points[-1]
            endpoint=self.get_end_points('constrained')[-1]
            v=endpoint-point
            distance=np.linalg.norm(v)
            for i in range(1,batch_size):
                next_points[i-1]=point+i/(batch_size)*v
            next_points[batch_size-1]=endpoint
        elif method=='pure_explore':
            next_points=self.choose_next_best_point_b(
                num_points,num_targets,angular_equivalence,sigma[0][0],
                delta,num_chosen=batch_size,exclusion=exclusion,
                allow_reduce=False,use_reduced_omega=True)
        elif method=='provided':
            next_points=n_points
        else:
            print('Error: unknown method')

        if next_points is None:
            return False

        if not rietveld_closest:
            start=len(self.points)
            self.points=np.append(self.points,next_points,axis=0)
            stop=len(self.points)
            for i in range(start,stop):
                self.create_line_from_sample_constrained(
                    self.constrained_dim,i,method='use_sigma',sigma=sigma)
            point_indexes=range(start,stop)
            self.update_values(point_indexes,sigma,scale,delta)
        else:
            closest_point=self.get_closest_point(next_points,index=False)
            self.points=np.append(self.points,[closest_point],axis=0)
            i = len(self.points)-1
            self.create_line_from_sample_constrained(
                self.constrained_dim,i,method='use_sigma',sigma=sigma)
            #self.update_values([i],sigma,scale,delta)
            self.make_p_gaussian(sigma,scale,delta)
            self.chosen_point=closest_point
        if return_points:
            return next_points

        if plot_process:
            pot_points=self.convert_points_to_new_projection(
                'berny',self.exploration_points)
            pot_targets=self.convert_points_to_new_projection(
                'berny',self.exploration_targets)
            points_t=self.convert_points_to_new_projection(
                'berny',self.points)
            end_points_t=self.get_end_points('berny')
            values=self.values/np.sum(self.values)
            post_data=self.convert_f_to_new_projection(
                'berny',values,self.omega)
            plotter.explore_post_data(
                pot_points,pot_targets,points_t,end_points_t,post_data)
            plotter.explore_plot("../figures/processfigures/explore")

        for point in next_points:
            if np.all(point==self.goal):
                return False
        return True

    def get_closest_point(self,next_points,index=True):
            mindist=9999
            closest_point_index=None
            closest_point=None
            goal=self.convert_to_standard_basis(self.goal)/self.cube_size
            for n,point in enumerate(next_points):
                spoint=self.convert_to_standard_basis(point)/self.cube_size
                dist=np.linalg.norm(spoint-goal)
                if dist<mindist:
                    mindist=dist
                    closest_point_index=n
                    closest_point=point
            if index:
                return closest_point_index
            else:
                return closest_point

    def get_closest_distance(self,next_points):
        chosen_point=self.get_closest_point(next_points,index=False)
        chosen_point=(self.convert_to_standard_basis(chosen_point)
                      /self.cube_size)
        goal=self.convert_to_standard_basis(self.goal)/self.cube_size
        return np.linalg.norm(goal-chosen_point)

    def get_max_individual_distance(self,next_points):
        chosen_point=self.get_closest_point(next_points,index=False)
        chosen_point=(self.convert_to_standard_basis(chosen_point)
                      /self.cube_size)
        goal=self.convert_to_standard_basis(self.goal)/self.cube_size
        return np.abs(goal-chosen_point).max()

    def revert_to_initial(self):
        self.points=self.points[0:1]
        self.lines=self.lines[0:1]
        self.end_points=self.end_points[0:1]

    def next_sample_test(self,sigma,scale,delta,num_lines):
        self.omega=self.omega_reduced
        self.values=self.values_reduced
        values_save=self.values
        scores=np.empty((len(self.omega),3))
        #scores=np.empty((len(self.values),3))
        tot=len(self.omega)
        for n,point in enumerate(self.omega):
            if not np.all(point == self.goal):
                print(tot-n)
                self.points=np.append(self.points,[point],axis=0)
                score=np.empty((num_lines,3))
                for i in range(num_lines):
                    self.create_line_from_sample_constrained(
                        self.constrained_dim,1,method='use_sigma',sigma=sigma)
                    self.update_values([1],sigma,scale,delta)
                    score[i]=self.get_score('mean_mode_variance')
                    self.values=values_save
                    self.lines=self.lines[0:1]
                    self.end_points=self.end_points[0:1]
                scores[n]=np.mean(score,axis=0)
                self.points=self.points[0:1]
            else:
                scores[n]=[0,0,0]
        self.end_points=self.get_end_points('constrained')
        self.create_plot_lines()
        plotter=visualise_square()
        self.heatmap=1+self.heatmap
        heatmap_save=self.heatmap
        meanscore=scores[:,0]/np.amax(scores[:,0])
        self.make_heatmap_constrained(meanscore,self.omega)
        plotter.next_sample_fig(
            self.heatmap,self.xlim,self.ylim,self.points,self.plot2d_lines,
            self.goal,"../figures/nextsampletest/mean.png")
        self.heatmap=heatmap_save
        meanscore=scores[:,1]/np.amax(scores[:,1])
        self.make_heatmap_constrained(meanscore,self.omega)
        plotter.next_sample_fig(
            self.heatmap,self.xlim,self.ylim,self.points,self.plot2d_lines,
            self.goal,"../figures/nextsampletest/max.png")
        self.heatmap=heatmap_save
        meanscore=scores[:,2]/np.amax(scores[:,2])
        self.make_heatmap_constrained(meanscore,self.omega)
        plotter.next_sample_fig(
            self.heatmap,self.xlim,self.ylim,self.points,self.plot2d_lines,
            self.goal,"../figures/nextsampletest/var.png")
            
    def setup_one_batch_on_line(self,k):
        normal_a=np.array(k['Normal a'])
        normal_b=np.array(k['Normal b'])
        cube_size=k['Cube size']
        delta_param=k['Delta param']
        scale=k['Scale']
        batch_size=k['Batch size']
        contained_point=np.array(k['Contained point'])
        sigma=k['Sigma']
        rietveld_closest=k['Rietveld closest']

        con_dim=normal_a.shape[0]-2
        delta=cube_size/delta_param
        normal_a=normal_a/np.linalg.norm(normal_a)
        normal_b=normal_b/np.linalg.norm(normal_b)
        normal_vectors=np.stack((normal_a,normal_b))
        contained_point=contained_point*cube_size/np.sum(contained_point)
        sigma=np.diag(np.array([sigma]*con_dim))

        self.setup(normal_vectors,contained_point,
                   cube_size,sigma)
        self.random_initialise(1)
        self.make_p_gaussian(sigma,scale,delta)
        self.sample_next_batch(
            'lastline',sigma,scale,delta,batch_size=batch_size,
            rietveld_closest=rietveld_closest)

    def setup_random(self,k):
        normal_a=np.array(k['Normal a'])
        normal_b=np.array(k['Normal b'])
        cube_size=k['Cube size']
        delta_param=k['Delta param']
        scale=k['Scale']
        contained_point=np.array(k['Contained point'])
        sigma=k['Sigma']
        max_points=k['Max points']

        num_points=random.randrange(2,max_points+1)
        con_dim=normal_a.shape[0]-2
        delta=cube_size/delta_param
        normal_a=normal_a/np.linalg.norm(normal_a)
        normal_b=normal_b/np.linalg.norm(normal_b)
        normal_vectors=np.stack((normal_a,normal_b))
        contained_point=contained_point*cube_size/np.sum(contained_point)
        sigma=np.diag(np.array([sigma]*con_dim))

        self.setup(normal_vectors,contained_point,
                   cube_size,sigma)
        self.random_initialise(num_points)
        self.make_p_gaussian(sigma,scale,delta)

    def setup_n_batches_recording(self,k):
        normal_a=np.array(k['Normal a'])
        normal_b=np.array(k['Normal b'])
        cube_size=k['Cube size']
        delta_param=k['Delta param']
        scale=k['Scale']
        contained_point=np.array(k['Contained point'])
        sigma=k['Sigma']
        rietveld_closest=k['Rietveld closest']
        num_batches=k['Number of batches']

        slope=k.get('Slope')
        intercept=k.get('Intercept')
        radius=k.get('Radius')
        centre=k['Centre']
        batch_size=k['Batch size']
        min_angle=k['Min angle']

        con_dim=normal_a.shape[0]-2
        delta=cube_size/delta_param
        normal_a=normal_a/np.linalg.norm(normal_a)
        normal_b=normal_b/np.linalg.norm(normal_b)
        normal_vectors=np.stack((normal_a,normal_b))
        contained_point=contained_point*cube_size/np.sum(contained_point)
        sigma=np.diag(np.array([sigma]*con_dim))

        self.closest_distances=np.zeros(num_batches)
        self.max_individual_distances=np.zeros(num_batches)

        self.setup(normal_vectors,contained_point,
                   cube_size,sigma)
        self.random_initialise(1)
        self.make_p_gaussian(sigma,scale,delta)
        self.sample_next_batch(
            'lastline',sigma,scale,delta,batch_size=batch_size,
            rietveld_closest=rietveld_closest)

        self.closest_distances[0]=self.get_closest_distance(self.points)
        self.max_individual_distances[0]=self.get_max_individual_distance(
            self.points)

        if k.get('Max'):
            batch_size-=1

        for i in range(1,num_batches):
            next_points=self.choose_next_best_points_sphere(
                centre,batch_size,min_angle,slope=slope,intercept=intercept,
                set_radius=radius)
            if k.get('Max'):
                f=self.values/np.sum(self.values)
                next_points=np.append(next_points,[self.get_max(f)],axis=0)
            self.closest_distances[i]=self.get_closest_distance(next_points)
            self.max_individual_distances[i]=self.get_max_individual_distance(
                next_points)

            for point in next_points:
                if np.all(point==self.goal):
                    return
            self.sample_next_batch(
                'provided',sigma,scale,delta,
                rietveld_closest=rietveld_closest,n_points=next_points)

    def setup_one_batch_on_line_rietveld(self,k):

        normal_a=np.array(k['Normal a'])
        normal_b=np.array(k['Normal b'])
        cube_size=k['Cube size']
        delta_param=k['Delta param']
        scale=k['Scale']
        batch_size=k['Batch size']
        contained_point=np.array(k['Contained point'])

        weights=k['Weights']
        weight_error=k['Weight error']
        formulas=k['Formulas']
        sampled_point=np.array(k['Sampled point'])
        goal_point=np.array(k['Goal point'])

        delta=cube_size/delta_param
        normal_a=normal_a/np.linalg.norm(normal_a)
        normal_b=normal_b/np.linalg.norm(normal_b)
        normal_vectors=np.stack((normal_a,normal_b))
        contained_point=contained_point*cube_size/np.sum(contained_point)
        self.create_omega_constrained(
            normal_vectors,cube_size,contained_point)
        sampled_point=cube_size*sampled_point/np.sum(sampled_point)
        goal_point=cube_size*goal_point/np.sum(goal_point)
        self.goal=self.convert_point_to_constrained(goal_point)
        wt_convert=wt_converter()
        error_propagate=error_propagator(
            normal_a.shape[0],cube_size,contained_point)
        moles,moles_error,formulas_standard=wt_convert.wt_to_moles(
            formulas,weights,weights_error=weight_error)
        error_propagate.set_moles_error(moles,formulas_standard,moles_error)
        merged_mean,merged_sigma=error_propagate.get_merged_balls_p(
            self.basis)
        sigma=self.add_first_sample(sampled_point,merged_mean,merged_sigma)
        sigma=np.diag([sigma]*2)
        self.calculate_p_from_samples(delta,scale)
        self.sample_next_batch(
            'lastline',sigma,scale,delta,
            rietveld_closest=True,batch_size=batch_size,)

    def setup_rietveld_balls(self,k):

        normal_a=np.array(k['Normal a'])
        normal_b=np.array(k['Normal b'])
        cube_size=k['Cube size']
        delta_param=k['Delta param']
        scale=k['Scale']
        batch_size=k['Batch size']
        contained_point=np.array(k['Contained point'])

        weights=k['Weights']
        weight_error=k['Weight error']
        formulas=k['Formulas']
        sampled_point=np.array(k['Sampled point'])
        goal_point=np.array(k['Goal point'])

        delta=cube_size/delta_param
        normal_a=normal_a/np.linalg.norm(normal_a)
        normal_b=normal_b/np.linalg.norm(normal_b)
        normal_vectors=np.stack((normal_a,normal_b))
        contained_point=contained_point*cube_size/np.sum(contained_point)
        self.create_omega_constrained(
            normal_vectors,cube_size,contained_point)
        sampled_point=cube_size*sampled_point/np.sum(sampled_point)
        goal_point=cube_size*goal_point/np.sum(goal_point)
        self.goal=self.convert_point_to_constrained(goal_point)
        wt_convert=wt_converter()
        error_propagate=error_propagator(
            normal_a.shape[0],cube_size,contained_point)
        moles,moles_error,formulas_standard=wt_convert.wt_to_moles(
            formulas,weights,weights_error=weight_error)
        error_propagate.set_moles_error(moles,formulas_standard,moles_error)
        merged_mean,merged_sigma=error_propagate.get_merged_balls_p(
            self.basis)
        sigma=self.add_first_sample(sampled_point,merged_mean,merged_sigma)
        sigma=np.diag([sigma]*2)
        self.calculate_p_from_samples(delta,scale)
        self.sample_next_batch(
            'lastline',sigma,scale,delta,
            rietveld_closest=True,batch_size=batch_size,)
        radius=k.get('Radius')
        min_angle=k['Min angle']
        centre=k['Centre']

        next_points=self.choose_next_best_points_sphere(
            centre,batch_size,min_angle,set_radius=radius)
        self.found_unknown=False

        for point in next_points:
            if np.all(point==self.goal):
                self.found_unknown=True
                return

        self.sample_next_batch(
            'provided',sigma,scale,delta,
            rietveld_closest=True,n_points=next_points)

    def test_ball_batch_closest_variance_many(self,k,result_descriptors):
        if self.found_unknown:
            return None
        con_dim=self.constrained_dim
        values=self.values/np.sum(self.values)
        radii_reduction=k.get('Radius reduction')
        radii=k.get('Radius')
        slope=k.get('Slope')
        intercept=k.get('Intercept')
        custom_radius=k.get('Custom radius')
        all_next_points=self.eval_next_best_points_sphere_radius(
            k['Centre'],k['Batch size']-1,k['Min angle'],
            radii_reduction=radii_reduction,radii=radii,slope=slope,
            intercept=intercept,custom_radius=custom_radius)
        p=None
        if k['Centre']=='max':
            p=self.get_max(f=values)
        elif k['Centre']=='mean':
            p=self.get_mean(f=values)
        else:
            print('error, unknown centre method')

        if custom_radius is not None:
            dists=[]
            for n,next_p in enumerate(all_next_points):
                next_points=np.append(next_p,[p],axis=0)
                dists.append(self.get_closest_distance(next_points))
            return np.array([dists])
        else:
            results=np.empty(
                (all_next_points.shape[0],len(result_descriptors)))
            for n,next_p in enumerate(all_next_points):
                next_points=np.append(next_p,[p],axis=0)

                result=[]
                for descriptor in result_descriptors:
                    if descriptor=='Variance':
                        result.append(self.f_score(values,method='variance'))
                    elif descriptor=='Radius reduction':
                        result.append(radii_reduction[n])
                    elif descriptor=='Closest distance':
                        result.append(self.get_closest_distance(next_points))
                    elif descriptor=='Radius':
                        result.append(radii[n])
                    else:
                        print('Error: unknown descriptor')
                results[n]= np.array(result)

        return results

    '''
    def test_ball_best(self,k,result_descriptors):
        evaluation_method=k['Evaluation method']
        radii=k['Radii']

        con_dim=self.constrained_dim
        values=self.values/np.sum(self.values)

        all_next_points=self.eval_next_best_points_sphere_radius(k['centre'],
            k['batch_size']-1,k['min angle'],k['Radius reduction'])

        p=None
        if k['centre']=='max':
            p=self.get_max(f=values)
        elif k['centre']=='mean':
            p=self.get_mean(f=values)
        else:
            print('error, unknown centre method')
        results=np.empty((all_next_points.shape[0],len(result_descriptors)))

        max_score=99999
        max_radius=None
        for n,next_p in enumerate(all_next_points):
            next_points=np.append(next_p,[p],axis=0)
            score=None
            if evaluation_method=='Closest distance':
                score=self.get_closest_distance(next_points)
            else:
                print('Error, unknown evaluation method')
            if score < max_score:
                max_score=score
                max_radius=radii[n]
                o
                o
            for descriptor in result_descriptors:
                if descriptor=='Variance':
                    result.append(self.f_score(values,method='variance'))
                elif descriptor=='Radius reduction':
                    result.append(k['Radius reduction'][n])
                elif descriptor=='Closest distance':
                    result.append(self.get_closest_distance(next_points))
                else:
                    print('Error: unknown descriptor')
            results[n]= np.array(result)
        return results
    '''

    def test_ball_batch_closest_variance(self,k,result_descriptors):
        con_dim=self.constrained_dim
        values=self.values/np.sum(self.values)
        radius_reduction=k.get('Radius reduction')
        intercept=k.get('Intercept')
        slope=k.get('Slope')
        radius=k.get('Radius')
        next_points=self.choose_next_best_points_sphere(
            k['Centre'],k['Batch size']-1,k['Min angle'],
            radius_reduction=radius_reduction,slope=slope,intercept=intercept,
            set_radius=radius)
        result=[]
        for descriptor in result_descriptors:
            if descriptor== 'Variance':
                result.append(self.f_score(values,method='variance'))
            elif descriptor== 'Radius reduction':
                result.append(k['Radius reduction'])
            elif descriptor== 'Closest distance':
                result.append(self.get_closest_distance(next_points))
            elif descriptor=='Max element % difference':
                print('Goal:',goal)
                print('Chosen:',chosen_point)
                maxdiff=0
                for e in range(len(chosen_point)):
                    diff=abs(chosen_point[e]-goal[e])
                    pdiff=diff/goal[e]
                    if pdiff > maxdiff:
                        maxdiff = pdiff
                result.append(maxdiff)
            else:
                print('Error - unknown result descriptor')

        return np.array([result])

    def test_num_batches_condition(self,k,result_descriptors):
        unfound=True
        sample_type=k['Sample type']
        condition_type=k['Condition type']
        condition_value=k['Condition values']
        rietveld_closest=k['Rietveld closest']
        scale=self.scale
        delta=self.delta
        sigma=self.sigma

        batches=0
        while(unfound):
            if batches>20:
                print('Warning: exceeded 20 batches')
            #test if condition is met
            val=None
            if condition_type=='Closest distance':
                val=self.get_closest_distance(self.points)
            else:
                print('Error: unknown condition type')
            if val<condition_value:
                unfound=False
            else:
                batches=batches+1
                next_points=None
                if sample_type=='ball':
                    values=self.values/np.sum(self.values)
                    centre=k['Centre']
                    minangle=k['Min angle']
                    radius_reduction=k['Radius reduction']
                    batch_size=k['Batch size']
                    next_points=self.choose_next_best_points_sphere(
                        centre,batch_size-1,minangle,radius_reduction)
                else:
                    print('Error: unknown sample type')

                self.sample_next_batch(
                    'provided',sigma,scale,delta,
                    rietveld_closest=True,n_points=next_points)

        results=[]
        for descriptor in result_descriptors:
            if descriptor=='Number of batches':
                results.append(batches)
        return np.array([results])

    def test(self,k,result_descriptors):
        values=self.values/np.sum(self.values)

        result=[]
        for d in result_descriptors:
            if d == 'Variance':
                result.append(self.f_score(values,method='variance'))
            elif d == 'Mean distance':
                mean=self.get_mean(f=self.values)
                result.append(np.linalg.norm(mean-self.goal))
            elif d == 'Max distance':
                centre=self.get_max(f=self.values)
                result.append(np.linalg.norm(centre-self.goal))
            elif d == 'Standard deviation':
                result.append(np.sqrt(self.f_score(values,method='variance')))
            elif d=='Variance from max':
                result.append(self.f_score(values,method='max_variance'))
            elif d=='Distance mean max':
                centre=self.get_max(f=self.values)
                mean=self.get_mean(f=self.values)
                result.append(np.linalg.norm(mean-centre))
            elif d=='Number samples':
                result.append(len(self.points))
            elif d == 'Batch number':
                results=np.empty((len(self.closest_distances),3))
                for n,(i,j) in enumerate(
                    zip(self.closest_distances,self.max_individual_distances)):
                    results[n]=np.array([n,i,j])
                #return early as batch number has other descriptors for making
                #dataframe (shouldnt be an issue but would print error)
                return results
            else:
                print('Error: unknown result descriptor')
                print(d)
        return np.array([result])














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
              #block
'''


