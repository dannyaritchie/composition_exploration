import ternary
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.ma as ma
from algorithm import *
from pathlib import Path
from wtconversion import *
from errorpropagator import * 
from dataparser import *
from results import *

#add test comment

def random_angle(upper, method):
    if method == 'uniform':
        return (np.pi/180)*round(np.random.uniform(-upper,upper),10)


def dim_test(points = 6, dim = 4, theta_range = 20,
             theta_distribution='uniform', batchsize=5):
    setup = all_information(dim=dim)
    setup.add_random_initial(dim)
    setup.add_random_goal(dim)
    setup.create_line_from_point_nd(dim,0,theta_range,theta_distribution)
    target_points = setup.sampleonline(batchsize,0)
    for i in range(target_points[0],target_points[1]):
        setup.create_line_from_point_nd(dim,i,theta_range,theta_distribution)
    setup.prod_theta_score_n(increment=5,return_type='with_index')
    setup.get_points_for_exploration_evaluation()
    setup.weight_points_for_exploration_evaluation()




def stepa(batchsize=5,show_image=10,degree_error=20,method='uniform',output_file=False,
     folder = 'visualisations/heatmap_many/', name =
          'test',distance_by_success=False,opt=True,brute=False):
    setup = all_information(dim=3)
    setup.add_random_initial(3)
    setup.add_random_goal(3)
    setup.create_line_from_point(0, random_angle(degree_error,method))
    setup.set_end_points()
    target_points = setup.sampleonline(batchsize,0)
    for i in range(target_points[0],target_points[1]):
        setup.create_line_from_point(i, random_angle(degree_error,method))
        setup.set_end_points()
    setup.set_labels()
    res = None
    if opt:
        res = setup.cgmo_angle()
    success = 'successful'
    if output_file:
        if res.success == False:
            success = 'unsuccessful'
            name += 'Failed'
        file =  open(folder + 'optimisation_result/' + name
                   +'.txt','w+')
        file.write(str(res))
        file.close()
    if show_image == 0:
        setup.plot(True,True,True,optimal_point=True,
                   show=False, angle_product_heatmap=True,
                   name=folder + name + '.eps', title =
                   'Heatmap of ADPF alongside optimal point, optimisation ' +
                   success + '\n' +
                   'batchsize = ' + str(batchsize) +', error = (Uniform, 20°)',
                    save=True)
    if show_image == 1:
        setup.plot(True,True,True,False,optimal_point=True,
                   show=True, angle_product_heatmap=False,
                   name='heatmap5lines.eps', title =
                   'Maximum of angle product after 1 step,\n' +
                   'batchsize = ' + str(batchsize) +', error = (Uniform, 20°)')
    if show_image == 2:
        setup.plot(True,True,True,optimal_point=True,
                   show=True, angle_product_heatmap=True,
                   name=folder + name + '.eps', title =
                   'Heatmap of ADPF alongside optimal point,\n' +
                   'batchsize = ' + str(batchsize) +', error = (Uniform, 20°)',
                    save=True)
    if brute:
        setup.prod_theta_score_n(dim = 3)
    if distance_by_success:
        return (res.success,setup.getdistance())
    if opt:
        return setup.getdistance()
   # setup.nllsq_angle()
   # setup.least_squares_solution()

def stepa_alt():
    distance = np.empty((4,100))
    for n,b in enumerate([3,5,7,9]):
        print(b)
        for j in range(100):
            initial_point = random_point()
            goal_point = random_point()
            setup = all_information(goal_point,initial_point)
            setup.create_line_from_point(0,random_angle(20,'uniform'))
            for i in range(b):
                setup.add_point([random_point()])
                setup.create_line_from_point(i+1,
                                             random_angle(20,'uniform'))
                setup.set_end_points()
            res = setup.cgmo_angle()
            distance[n][j] = setup.getdistance()
    return distance





# Make images higher resolution and set default size
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (4, 4)

#code for making many heatmap, optimal point pairs
#crashes for hist
'''
for i in range(100):
    b = 5
    stepa(batchsize = b, show_image=0, output_file=True, name =
                        str(i) + '_' + str(b))
'''


#code to plot dist vs batchsize
'''
bmax = 10
bmin = 3
distances = np.empty((4,100))
for n,b in enumerate([3,5,7,9]):
    for i in range(100):
        d = stepa(batchsize = b)
        distances[n][i] = d
        print(b,i,d)
'''
#code for plotting dist vs batchsize for random points
'''
distances = stepa_alt()
'''
#code for plotting the above two
'''
means = np.mean(distances,axis=1)
std = np.std(distances,axis=1)
fig, ax = plt.subplots()
ax.errorbar([3,5,7,9],means,std,ecolor='red',ls='', marker='x')
plt.ylabel('Mean distance between target point and maximum of ADPF',fontsize=5)
plt.xlabel('Batchsize',fontsize = 10)
fig.suptitle('Mean distance as a function of batchsize,\n points are chosen '
             'randomly, error = [Uniform,20]',fontsize=10)
plt.show()
'''
#code to plot histogram
'''
suc_distance = []
unsuc_distance = []
b = 5
for i in range(500):
    print(i)
    succ,dist = stepa(batchsize = b,distance_by_success=True)
    if succ:
        suc_distance.append(dist)
    else:
        unsuc_distance.append(dist)
fig, ax = plt.subplots()
ax.hist(suc_distance, bins=60)
plt.xlabel('Distance between target point and maximum of ADPF',fontsize=5)
fig.suptitle('Histogram for batchsize = ' +str(b)+', error = [uniform,20]' +
             '\nFor successful optimisation',fontsize=10)
plt.show()
plt.clf()
fig, ax = plt.subplots()
ax.hist(unsuc_distance, bins=60)
plt.xlabel('Distance between target point and maximum of ADPF',fontsize=5)
fig.suptitle('Histogram for batchsize = ' +str(b)+', error = [uniform,20]' +
             '\nFor unsuccessful optimisation',fontsize=10)
plt.show()
'''

#code for investigating heatmap time 
'''
dim_test()
'''
#code for testing f score
'''
setup = all_information()
setup.f_score_test()
'''
#code for plotting m,n vs final dist for differnt a
'''
theta_distribution='uniform'
theta_range=20
increment=1
angular_equivalence=20
name='../data/evaluation_opt/3d/process/'
batch_size=3
k=np.pi
#f.close()
p = Path('../data/evaluation_opt/distancesa.txt')
if not p.is_file():
    print('hi')
    f = p.open(mode = 'w')
    f.write('dim' + ' ' + 'number_points' + ' ' + 'number_targets' + ' ' +
            'theta_range' + ' ' + 'theta_distribution' + ' ' +
            'angular_equivalence' + ' ' + 'increment' + ' ' +
            'fractional_cutoff' + ' ' + 'batch_size' + ' ' + 'k' + ' ' +
            'distance' + '\n')
    f.close()
for i in range(1):
    for dim in [3]:
        for a in [0.5]:
            namea=name + str(a)
            for m in [50]:
                nameam = namea + '_' + str(m)
                for n in [20]:
                    print(i)
                    nameamn = nameam + '_' + str(n) + '_' + str(i) + '.eps'
                    nameamn=''
                    setup=all_information()
                    distance=setup.evaluation(dim,n,m,theta_range,theta_distribution,angular_equivalence,increment,a,batch_size,k,nameamn,plot_process=True)
                    f=p.open(mode = 'a')
                    f.write(str(dim) + ' ' + str(n) + ' ' + str(m) + ' ' +
                            str(theta_range) + ' ' + theta_distribution + ' ' +
                            str(angular_equivalence) + ' ' + str(increment) + ' ' +
                            str(a) + ' ' + str(batch_size) + ' ' + str(k) + ' ' +
                            str(distance) + '\n')
                    f.close()
'''
#code for testing charge constraint
'''
setup=all_information()
normal_a = np.array([1,2,-2,-1,-1])
normal_b = np.array([1,1,1,1,1])
normal_a = normal_a/np.linalg.norm(normal_a)
normal_b = normal_b/np.linalg.norm(normal_b)
normal_vectors=np.stack((normal_a,normal_b),axis=0)
cube_size=20
contained_point=np.array([1/3,1/6,1/6,1/6,1/6])*cube_size
setup.charge_constraint_test(5,normal_vectors,cube_size,contained_point,20,'uniform',4)
'''
#code for testing linearly constrained omega creation
'''
setup=all_information()
normal_a = np.array([1,2,-2,-1,-1])
#Cs,Bi,X,I,Cl
normal_b = np.array([1,1,1,1,1])
normal_a = normal_a/np.linalg.norm(normal_a)
normal_b = normal_b/np.linalg.norm(normal_b)
normal_vectors=np.stack((normal_a,normal_b),axis=0)
cube_size=10
contained_point=np.array([1/3,1/6,1/6,1/6,1/6])*cube_size
setup.create_omega_constrained(5,normal_vectors,cube_size,contained_point)
setup.convert_to_standard_basis()
setup.plot_jon_test()
'''
#code for testing pawley simulation
'''
setup=all_information()
normal_a = np.array([1,2,-2,-1,-1])
#Cs,Bi,X,I,Cl
normal_b = np.array([1,1,1,1,1])
normal_a = normal_a/np.linalg.norm(normal_a)
normal_b = normal_b/np.linalg.norm(normal_b)
normal_vectors=np.stack((normal_a,normal_b),axis=0)
cube_size=25
contained_point=np.array([1/3,1/6,1/6,1/6,1/6])*cube_size
setup.create_omega_constrained(5,normal_vectors,cube_size,contained_point)
setup.simulate_pawley()
setup.incorporate_pawley()
'''
#code for plotting intial pawley information
'''
setup=all_information()
normal_a = np.array([1,2,-2,-1,-1])
#Cs,Bi,X,I,Cl
normal_b = np.array([1,1,1,1,1])
normal_a = normal_a/np.linalg.norm(normal_a)
normal_b = normal_b/np.linalg.norm(normal_b)
normal_vectors=np.stack((normal_a,normal_b),axis=0)
cube_size=10
contained_point=np.array([1/3,1/6,1/6,1/6,1/6])*cube_size
setup.create_omega_constrained(5,normal_vectors,cube_size,contained_point)
setup.simulate_pawley()
setup.plot_pawley_test()
'''

'''
#code for figure generation 
setup=all_information()
setup.add_initial(3)
setup.add_point([[40,40,20]])
setup.add_point([[25,25,50]])
setup.add_goal([20,50,30])
setup.lines = np.empty((0,3))
setup.end_points = np.empty((0,3))
for i in range(3):
    setup.create_line_from_point_nd(3,i,20,'uniform')
setup.set_labels(method='custom_figure')
setup.plot(True,True,True,angle_product_heatmap=True)
'''
#code for testing wt% conversion
'''
wt_converter = wt_converter()
formula1="H 12"
formula2="C 6"
f1="Cs 3 Bi 2 I 9"
f2="Cs I"
f3="Bi 2 Se 3"
wt_converter.wt_to_moles([f1,f2,f3],[0.4,0.5,0.1])
#code for testing error propagator
'''
'''
p=error_propagator()
p.test_2d()
'''
#code for testing line from sample
'''
setup=all_information()
normal_a = np.array([1,2,-2,-1,-1])
#Cs,Bi,X,I,Cl
normal_b = np.array([1,1,1,1,1])
normal_a = normal_a/np.linalg.norm(normal_a)
normal_b = normal_b/np.linalg.norm(normal_b)
normal_vectors=np.stack((normal_a,normal_b),axis=0)
cube_size=10
contained_point=np.array([1/3,1/6,1/6,1/6,1/6])*cube_size
setup.create_line_from_sample_test_uncon()
'''
#code for testing pawley simulation (with arbitrary ranking)
'''
setup=all_information()
normal_a = np.array([1,3,-2,-1,-1])
#Cs,Bi,X,I,Cl
normal_b = np.array([1,1,1,1,1])
#normal_a = normal_a/np.linalg.norm(normal_a)
#normal_b = normal_b/np.linalg.norm(normal_b)
normal_vectors=np.stack((normal_a,normal_b),axis=0)
cube_size=15
contained_point=np.array([1/3,1/6,1/6,1/6,1/6])*cube_size
setup.create_omega_constrained(5,normal_vectors,cube_size,contained_point)
setup.set_pawley_rank()
setup.plot_pawley_ranking()
setup.incorporate_pawley(kind='arbitrary_rank')
'''
#code for testing data parser
'''
data = jon()
for key in data.sampled_points_dict:
    print(data.sampled_points_dict[key])
'''
#code for creating square
'''
setup=all_information()
normal_a = np.array([1,3,-2,-1])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=70
contained_point=np.array([1,1,1,2])*cube_size/5
setup.create_omega_constrained(4,normal_vectors,cube_size,contained_point)
setup.plot_square_test()
'''
#code for making apropriate sigma
'''
sigmas=np.empty((100,2))
for k in range(100):
    print(k)
    setup=all_information()
    normal_a = np.array([1,3,-2,-1])
    normal_b = np.array([1,1,1,1])
    normal_vectors=np.stack((normal_a,normal_b)) 
    cube_size=10
    contained_point=np.array([1,1,1,2])*cube_size/5
    setup.create_omega_constrained(4,normal_vectors,cube_size,contained_point)
    formula_a="Cs 5 Bi 1 Se 1 I 6"
    formula_d="Cs 5 Bi 1 Se 3 I 2"
    formula_b="Cs 1 Bi 5 Se 3 I 10"
    formula_c="Cs 1 Bi 5 Se 7 I 2"
    formulas=[formula_a,formula_b,formula_c,formula_d]
    weights=[0.25,0.25,0.25,0.25]
    wt_convert=wt_converter()
    error_propagate=error_propagator(dim=4)
    moles,moles_error,formulas_standard=wt_convert.wt_to_moles(formulas,weights)
    error_propagate.set_moles_error(moles,formulas_standard,moles_error)
    merged_ball=error_propagate.get_merged_balls_p(setup.basis)
    sigmas[k]=merged_ball[1]
    for i,j in zip(merged_ball[0],merged_ball[1]):
        print(i,'mean')
        print(j,'sigma')
    p = Path('../testdata/apropriatesigma.txt')
    if not p.is_file():
        print('hi')
        f = p.open(mode = 'a')
        f.write('-----------\n')
        f.write('Known crystals:\n')
        for i in formulas:
            f.write('\t'+i+'\n') 
        f.write('Weights:\n')
        f.write('\t weights\n')
        f.write('Errors\n')
        f.write('\t Default\n')
        f.write('\t (0.03 on all weights, 0.5g total mass, 2% total mass error\n')
        f.write('Sigmas:')
        f.close()
    p = Path('../testdata/apropriatesigma.txt')
    f = p.open(mode = 'a')
    f.write('\t'+str(merged_ball[1])+'\n')
    f.close()
avg_sigma=np.mean(sigmas,axis=0)
p = Path('../testdata/apropriatesigma.txt')
f = p.open(mode = 'a')
f.write('Average sigma:\n')
f.write('\t'+str(avg_sigma)+'\n')
f.close()
#Result summary - 0.0188
'''
#code for testing line from sample in constrained
'''
setup=all_information()
normal_a = np.array([1,3,-2,-1])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=70
contained_point=np.array([1,1,1,2])*cube_size/5
sigma=np.diag(np.array([0.0188,0.0188])/0.1)
setup.create_line_from_sample_test(normal_vectors,cube_size,contained_point,sigma)
'''
#code for testing 2d heatmap
'''
setup=all_information()
normal_a = np.array([1,3,-2,-1])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=500
contained_point=np.array([1,1,1,2])*cube_size/5
dist=0.5 #simulated distance of avg known composition to sample
sigma=np.diag(np.array([0.0188,0.0188])/dist)
setup.heatmap_test(
    normal_vectors,cube_size,contained_point,sigma)
'''
#code for finding next points (with jon pawley a)
'''
setup=all_information()
normal_a = np.array([1,3,-2,-1,-1])
test=np.array([1,1,1,1,1])
print(np.dot(test,normal_a))
#Cs,Bi,X,I,Cl
normal_b = np.array([0.60,7.24,0.98,13.51,6.84])
print(np.dot(test,normal_b))
#normal_a = normal_a/np.linalg.norm(normal_a)
#normal_b = normal_b/np.linalg.norm(normal_b)
normal_vectors=np.stack((normal_a,normal_b),axis=0)
cube_size=25
contained_point=np.array([1/3,1/6,1/6,1/6,2/6])*cube_size
setup.create_omega_constrained(5,normal_vectors,cube_size,contained_point)
pawley_rank=[[[1,6,5,4,5]],
             [[2,1,1,2,1]],
             [[2,1,1,1.5,1.5]],
             [[2,1,1,1,2]],
             [[1,4,5,1,2]]]
             #[[2,1,1,0,3],[2,1,1,3,0],[2,1,0,2,3]]]
#              [0,1,1,1,2],[2,0,1,1,2],[1,1,1,2,0],
#              [2,3,5,1,1]]]
setup.set_pawley_rank_s(pawley_rank=pawley_rank)
#setup.plot_pawley_ranking()
setup.incorporate_pawley(kind='arbitrary_rank',plot=False)
points=setup.get_uniform_from_pawley(plot=False,method='reduced_omega')
print('a',len(points))
p = Path('../testdata/suggestednextpoints.txt')
if not p.is_file():
    f = p.open(mode = 'a')
    f.write('Points:\n')
    f.close()
f = p.open(mode = 'a')
f.write('Points:\n')
for n,i in enumerate(points):
    print('Point ' + str(n) +': ',i)
    print(np.dot(i,normal_vectors[0]))
    f.write('\t'+str(i)+'\n')
f.close()
pawley_rank=[[[1,6,5,4,5]],
             [[2,1,1,2,1]],
             [[2,1,1,1.5,1.5]],
             [[2,1,1,1,2]],
             [[1,4,5,1,2]],
             [[2,1,1,0,3],[2,1,1,3,0],[2,1,0,2,3],
              [0,1,1,1,2],[2,0,1,1,2],[1,1,1,2,0],
              [2,3,5,1,1]]]
setup.set_pawley_rank_s(pawley_rank=pawley_rank)
#jon_points=[[3.11,7.33,8.72,7.66,0],[2.30,8.15,9.20,6.18,2.17],[7.71,8.62,9.05,8.43,7.04],[0.80,13.16,13.41,13.11,0.35],[2.35,6.83,8.67,3.86,1.64]]
jon_points=[[2.21,8.22,2.28,13.28,9.03],[1.07,5.44,0,11.09,5.61],[3.13,4.98,2.84,7.23,5.16],[1.83,5.22,1.20,8.05,7.04],[0.60,7.24,0.98,13.51,6.84]]
setup.plot_points_jon(pawley_rank=True,points=jon_points)
'''
#code for testing berny plotting
'''
setup=all_information()
normal_a = np.array([1,2,-2,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
contained_point=np.array([1,1,1,2])*cube_size/5
dist=0.5 #simulated distance of avg known composition to sample
sigma=np.diag(np.array([0.0188,0.0188])/dist)
setup.berny_test(normal_vectors,cube_size,contained_point,sigma,method='angle')
'''
#code for saving results
'''
directory='../data/a/'
results=Results(directory)
results.two_d_results('test.csv',m=1000)
'''
#code for plotting results
'''
selection_dic={
    'Score method' : 'd_g_mu',
    'P method' : 'guassian',
    'std' : 0.0188*2,
    'cube_size' : 100,
    'Phase field key' : 'A',
    #'Parameter key' : 'A',
    'Number of points': 4}
directory='../data/a/'
results=Results(directory)
results.plot_mean_vs('Parameter key',selection_dic,'test.csv',
                    save_path='../results/a/mu_p_3.png')
'''
#code for getting mean for adding points
'''
directory='../data/adding_points_scores/'
results=Results(directory)
results.get_score_vs_add_points()
'''
#code for plottting adding point scores
directory='../data/adding_points_scores/'
results=Results(directory)
results.plot_score_vs_add_point()

