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
import seaborn as sns

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
cube_size=300
contained_point=np.array([1,1,1,2])*cube_size/5
sigma=np.diag(np.array([0.3,0.3]))
setup.setup(normal_vectors,contained_point,cube_size,sigma,create_heatmap=True)
setup.random_initialise(2)
setup.make_p_gaussian(sigma,1,1)
plotter=visualise_square()
setup.create_plot_lines()
setup.make_heatmap_constrained()
plotter.test_heat(setup.points,setup.plot2d_lines,setup.heatmap,setup.xlim,setup.ylim)
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
contained_point=np.array([2,1,1,1])*cube_size/5
dist=0.05 #simulated distance of avg known composition to sample
sigma=np.diag(np.array([0.0188,0.0188])/dist)
setup.berny_test(normal_vectors,cube_size,contained_point,sigma,method='p')
'''
#code for saving results
'''
directory='../data/a/'
results=Results(directory)
results.two_d_results('unsure.csv',m=1000)
'''
#code for plotting results
'''
selection_dic={
    'Score method' : 'd_g_mu',
    'P method' : 'angle_product',
    'std' : 0.0188*2,
    'cube_size' : 100,
    'Phase field key' : 'A',
    #'Parameter key' : 'A',
    'Number of points': 3}
directory='../data/a/'
results=Results(directory)
results.plot_mean_vs('Parameter key',selection_dic,'unsure.csv',
                    save_path='../results/a/mu_p_3.png')
'''
#code for getting mean for adding points
'''
directory='../data/adding_points_scores/'
filename='testtest.txt'
#filename='s1_d1_p2_a.txt'
results=Results(directory)
results.get_score_vs_add_points(filename)
'''

#code for plottting adding point scores
'''
setup=all_information()
normal_a = np.array([1,2,4,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
contained_point=np.array([2,1,1,4])*cube_size/8
sigma=np.diag([0.186,0.186])
scale=1
delta=1
power=1
num_samples=4
directory='../figures/processfigures/'
setup.setup(normal_vectors,contained_point,cube_size,sigma,create_heatmap=True)
setup.random_initialise(1)
points=setup.convert_points_to_new_projection('berny',setup.points)
goal=setup.convert_points_to_new_projection('berny',setup.goal)
setup.make_p_gaussian(sigma,scale,delta)
plotter=Plotter('bernyterny')
plotter.process_a(goal,num_samples)
labels={
    'Initial':[0,1],
}
for i in range(num_samples):
    print(i)
    end_points=setup.get_end_points('berny')
    unsuc = setup.sample_next_point(
        'a',sigma,power=power)
    points=np.append(points,setup.convert_points_to_new_projection(
        'berny',np.array([setup.points[-1]])),axis=0)
    values_norm=setup.values/np.sum(setup.values)
    data=setup.convert_f_to_new_projection('berny',values_norm,setup.omega)
    filename=str(i)+'.png'
    plotter.berny_heat_points(directory+filename,data,points,labels,end_points)
    labels['Sample ' + str(i)]=[i+1,i+2]
    labels['Next sample point']:[i+2,i+3]
    setup.update_values([-1],sigma,scale,delta)
'''






#code for making figures for presentations
    #Figure of multiple small balls
'''
setup=all_information()
normal_a = np.array([1,2,4,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
contained_point=np.array([2,1,1,4])*cube_size/8
setup.create_omega_constrained(
    normal_vectors,cube_size,contained_point,create_heatmap=True)
f1='Li 4 Zn 0 Si 1 S 4'
f2='Li 0 Zn 1 Si 0 S 1'
f3='Li 2 Zn 0 Si 0 S 1'
sampled_point=np.array([24,10,9,40])
sampled_point=cube_size*sampled_point/np.sum(sampled_point)
formulas=[f1,f2,f3]
weights=[0.47,0.36,0.07]
wt_convert=wt_converter()
error_propagate=error_propagator(4,cube_size,contained_point)
moles,moles_error,formulas_standard=wt_convert.wt_to_moles(
    formulas,weights,weights_error=[0.05])
error_propagate.set_moles_error(moles,formulas_standard,moles_error)
merged_mean,merged_sigma=error_propagate.get_merged_balls_p(setup.basis)
(mean,sigma)=error_propagate.get_merged_balls()
print(mean)
print(sigma)
small_means,small_sigmas=error_propagate.get_small_balls_p(setup.basis)
#setup.plot_small_balls(small_means,small_sigmas,projection='None')
point_labels=['Li4SiS4','ZnS','Li2S']
sample_label=['Li24Zn10Si9S40']
mean_label='K'
#setup.plot_small_balls(
    #small_means,small_sigmas,sampled_point,point_labels,sample_label,
    #projection='berny',heatmap=False)
setup.plot_merged_ball(
    merged_mean,merged_sigma,small_means,point_labels,mean_label,
    projection='berny')

#setup.plot_merged_ball(merged_mean,merged_sigma,projection='None')
#setup.add_first_sample(sampled_point,merged_mean,merged_sigma)
#delta=0.1
#scale=1
#setup.calculate_p_from_samples(delta,scale)
#setup.plot_p()
#setup.plot_merged_ball_p(merged_mean,merged_sigma,delta)
'''
#code for getting sigma
'''
sigmas=np.empty((200))
for i in range(1):
    setup=all_information()
    normal_a = np.array([1,2,4,-2])
    normal_b = np.array([1,1,1,1])
    normal_vectors=np.stack((normal_a,normal_b)) 
    cube_size=100
    contained_point=np.array([2,1,1,4])*cube_size/8
    setup.create_omega_constrained(
        normal_vectors,cube_size,contained_point,create_heatmap=True)
    f1='Li 4 Zn 0 Si 1 S 4'
    f2='Li 0 Zn 1 Si 0 S 1'
    f3='Li 2 Zn 0 Si 0 S 1'
    sampled_point=np.array([24,10,9,40])
    sampled_point=cube_size*sampled_point/np.sum(sampled_point)
    formulas=[f1,f2,f3]
    weights=[0.5,0.39,0.10]
    wt_convert=wt_converter()
    error_propagate=error_propagator(4,cube_size,contained_point)
    moles,moles_error,formulas_standard=wt_convert.wt_to_moles(
        formulas,weights,weights_error=[0.05])
    print(moles/np.sum(moles)*6)
    print(6*moles_error/np.sum(moles))
    error_propagate.set_moles_error(moles,formulas_standard,moles_error)
    print(error_propagate.get_small_balls()[0])
    print(error_propagate.get_merged_balls()[0])
    merged_mean,merged_sigma=error_propagate.get_merged_balls_p(setup.basis)
    #setup.plot_merged_ball(merged_mean,merged_sigma,projection='None')
    sigma=setup.add_first_sample(sampled_point,merged_mean,merged_sigma)[0]
    sigmas[i]=sigma[0]
    sigmas[100+i]=sigma[1]
    #delta=0.1
    #scale=1
    #setup.calculate_p_from_samples(delta,scale)
    #setup.plot_p()
    #setup.plot_merged_ball_p(merged_mean,merged_sigma,delta)
print(np.mean(sigmas))
#code for testing batch sampling
#1) sample along initial line vs use exploration
'''
#sample along line plot
'''
setup=all_information()
normal_a = np.array([1,2,4,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
contained_point=np.array([2,1,1,4])*cube_size/8
sigma=np.diag([0.186,0.186])
scale=1
delta=1
power=1
num_batches=1
batch_size=5
directory='../figures/processfigures/sampleonline'
setup.setup(normal_vectors,contained_point,cube_size,sigma,create_heatmap=True)
setup.random_initialise(1)
goal=setup.convert_points_to_new_projection('berny',setup.goal)
setup.make_p_gaussian(sigma,scale,delta)
plotter=Plotter('bernyterny')
plotter.process_batch(goal,num_batches)
labels={
    'Initial':[0,1],
    'First batch':[1,6]
}
for i in range(num_batches):
    print(i)
    setup.sample_next_batch('lastline',sigma,scale,delta,batch_size=batch_size,)
    points=setup.convert_points_to_new_projection('berny',setup.points)
    end_points=setup.get_end_points('berny')
    values_norm=setup.values/np.sum(setup.values)
    data=setup.convert_f_to_new_projection('berny',values_norm,setup.omega)
    filename=str(i)+'.png'
    plotter.berny_heat_points(directory+filename,data,points,labels,end_points)
'''
#sample along line vs explore score
'''
num_trials=1000
setup=all_information()
normal_a = np.array([1,2,4,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
contained_point=np.array([2,1,1,4])*cube_size/8
sigma=np.diag([0.186,0.186])
scale=1
delta=1
power=1
batch_size=5
num_points=50
num_targets=50
angular_equivalence=10
exclusion=3
directory='../figures/sampleonlinevsexploration/'
scores=np.empty((num_trials,3))
scoresb=np.empty((num_trials,3))
for i in range(num_trials):
    print(i,'--------')
    setup.setup(normal_vectors,contained_point,cube_size,sigma)
    setup.random_initialise(1)
    setup.make_p_gaussian(sigma,scale,delta,save_reduced=True)
    if setup.sample_next_batch(
        'pure_explore',sigma,scale,delta,batch_size=batch_size,
        num_points=num_points,num_targets=num_targets,
        angular_equivalence=angular_equivalence,exclusion=exclusion):
        scoresb[i]=setup.get_score('mean_mode_variance')
    else:
        scoresb[i]=[0,0,0]
    setup.revert_to_initial()
    setup.make_p_gaussian(sigma,scale,delta,save_reduced=True)
    if setup.sample_next_batch('lastline',sigma,scale,delta,
    batch_size=batchsize):
        scores[i]=setup.get_score('mean_mode_variance')
    else:
        scoresb[i]=[0,0,0]
dscores=scores-scoresb
ys = ['Distance from mean','Distance from max','Variance']
df=pd.DataFrame(
    data=scores,columns=ys)
dfb=pd.DataFrame(
    data=scoresb,columns=ys)
dfc=pd.DataFrame(
    data=dscores,columns=ys)
dfb['Method']='Explore A'
df['Method']='On line'
dfc['Method']='On line - Explore'
print(dfb[dfb['Method']=='Explore A'].shape[0])
print(df[df['Method']=='On line'].shape[0])
df=df.append(dfb)
print(df.shape[0])
for y in ys:
    ax=sns.pointplot(
        x='Method',y=y,data=df,capsize=1,linestyles="",errwidth=1)
    plt.savefig(directory+y + '.png',bbox_inches='tight')
    plt.clf()
    ax=sns.pointplot(
        x='Method',y=y,data=dfc,capsize=1,linestyles="",errwidth=1)
    plt.savefig(directory+y + '_diff.png',bbox_inches='tight')
    plt.clf()
path="../data/sampleonlinevsexploration/a.csv"
df.to_csv(
    path,mode='a',header=not os.path.exists(path),index=False)
'''
#plot process for explore
'''dd
setup=all_information()
normal_a = np.array([1,2,4,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
contained_point=np.array([2,1,1,4])*cube_size/8
sigma=np.diag([0.186,0.186])
scale=1
delta=1
power=1
batchsize=5
num_points=50
num_targets=50
angular_equivalence=10
exclusion=3
directory='../figures/sampleonlinevsexploration/'
setup.setup(normal_vectors,contained_point,cube_size,sigma)
setup.random_initialise(1)
setup.make_p_gaussian(sigma,scale,delta,save_reduced=True)
setup.sample_next_batch(
    'pure_explore',sigma,scale,delta,batch_size=batchsize,
    num_points=num_points,num_targets=num_targets,
    angular_equivalence=angular_equivalence,
    exclusion=exclusion,plot_process=True)
'''
#code to test next sample algorithm
'''
setup=all_information()
normal_a = np.array([1,2,4,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
contained_point=np.array([2,1,1,4])*cube_size/8
sigma=np.diag([0.186,0.186])
scale=1
delta=1
power=1
num_lines=500
directory='../figures/sampleonlinevsexploration/'
setup.setup(normal_vectors,contained_point,cube_size,sigma,create_heatmap=True)
setup.random_initialise(1)
setup.make_p_gaussian(sigma,scale,delta,save_reduced=True)
setup.next_sample_test(sigma,scale,delta,num_lines)
'''
#code for optimising batch on line parameters for jon field
'''
setup=all_information()
normal_a = np.array([1,3,-1,-1,-2])
normal_b = np.array([1,1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=40
contained_point=np.array([1,1,1,1,1])*cube_size/5
sigma=np.diag([0.186,0.186,0.186])
scale=2
delta_param=20
delta=cube_size/delta_param
num_trials=1000
scores=np.empty((num_trials,3))
for i in range(num_trials):
    print(i)
    setup.setup(normal_vectors,contained_point,cube_size,sigma)
    setup.random_initialise(1)
    setup.make_p_gaussian(sigma,scale,delta)
    if setup.sample_next_batch('lastline',sigma,scale,delta,batch_size=5):
        scores[i]=setup.get_score('mean_mode_variance')
    else:
        scores[i]=n.array([0,0,0])
ys = ['Distance from mean','Distance from max','Variance']
df=pd.DataFrame(
    data=scores,columns=ys)
df['delta_param']=delta_param
df['sigma']=sigma[0][0]
df['scale']=scale
directory='../figures/jons1opt/'
path="../data/jons1opt/c.csv"
df.to_csv(
    path,mode='a',header=not os.path.exists(path),index=False)
'''
#code for plotting
'''
path="../data/jons1opt/c.csv"
df=pd.read_csv(path)
ys = ['Distance from mean','Distance from max','Variance']
directory='../figures/jons1opt/delta_'
for y in ys:
    ax=sns.pointplot(
        x='delta_param',y=y,data=df,capsize=1,linestyles="",errwidth=1)
    plt.savefig(directory+y + '.png',bbox_inches='tight')
    plt.clf()
'''
'''
    ax=sns.pointplot(
        x='Method',y=y,data=dfc,capsize=1,linestyles="",errwidth=1)
    plt.savefig(directory+y + '_diff.png',bbox_inches='tight')
    plt.clf()
'''
#code for plotting matt presentation figs
'''
setup=all_information()
normal_a = np.array([1,2,4,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
delta_param=2
scale=1
delta=cube_size/delta_param
contained_point=np.array([2,2,1,5])*cube_size/10
setup.create_omega_constrained(
    normal_vectors,cube_size,contained_point,create_heatmap=True)
f1='Li 4 Zn 0 Si 1 S 4'
f2='Li 0 Zn 1 Si 0 S 1'
f3='Li 2 Zn 0 Si 0 S 1'
sampled_point=np.array([24,10,9,40])
goal_point=np.array([2,1,1,4])
sampled_point=cube_size*sampled_point/np.sum(sampled_point)
goal_point=cube_size*goal_point/np.sum(goal_point)
setup.goal=setup.convert_point_to_constrained(goal_point)
formulas=[f1,f2,f3]
weights=[0.51,0.40,0.09]
wt_convert=wt_converter()
error_propagate=error_propagator(4,cube_size,contained_point)
moles,moles_error,formulas_standard=wt_convert.wt_to_moles(
    formulas,weights,weights_error=[0.05])
print(6*moles_error/np.sum(moles))
print(6*moles/np.sum(moles))
error_propagate.set_moles_error(moles,formulas_standard,moles_error)
merged_mean,merged_sigma=error_propagate.get_merged_balls_p(setup.basis)
small_means,small_sigmas=error_propagate.get_small_balls_p(setup.basis)
print(merged_sigma,'hhh')
sigma=setup.add_first_sample(sampled_point,merged_mean,merged_sigma)
print('!!!',sigma)
#delta=0.1
#scale=1
setup.calculate_p_from_samples(delta,scale)
plotter=Plotter('berny')
plotter.set_scatter_kwargs()
plotter.set_heat_cbar_kwargs()
plotter.set_directory('../../mat presentation figures/')
points=setup.convert_points_to_new_projection('berny',setup.points)
mean=setup.convert_points_to_new_projection('berny',merged_mean)
end_points=setup.get_end_points('berny')
#plotter.mean_line(points,end_points,mean)
small_means=setup.convert_points_to_new_projection('berny',small_means)
labels=['Li$_4$SiS (1.88M)','ZnS (2.79M)','Li$_2$S (1.33M)']
#plotter.mean_small(mean,small_means,labels)
labels=['Li$_4$SiS','ZnS','Li$_2$So']
gl='Li$_2$ZnSiS$_4$'
goal=setup.convert_points_to_new_projection('berny',setup.goal)
#plotter.mean_all(
#    points[0],'Li$_24$Zn$_10$Si$_9$S$_40$',small_means,labels,goal,gl)
p=setup.make_merged_ball_values(merged_mean,merged_sigma)
data=setup.convert_f_to_new_projection('berny',p,setup.omega)
#plotter.merged_ball(data,mean)
data=setup.convert_f_to_new_projection('berny',setup.values,setup.omega)
#d0=setup.make_heatmap_constrained()
#plotter.p_mean_initial(data,mean,points)
sigma=np.diag([sigma]*2)
n_points=setup.sample_next_batch(
    'lastline',sigma,scale,delta,rietveld_closest=True,
    batch_size=5,return_points=True)
n_points=np.append(setup.points[0:1],n_points,axis=0)
#print(sampled_point)
#print(points)
points=setup.convert_points_to_new_projection('berny',n_points)
points_s=setup.convert_to_standard_basis(n_points)
phase_field=['Li','Zn','Si','S']
labels=['Initial']
colors=['Blue','darkgreen','olivedrab','mediumseagreen','springgreen','lime']
#print(points)
#print('a')
for i in points_s[1:]:
    norm=i/np.sum(i)
    label=''
    for j,e in zip(norm,phase_field):
        num=round(j,2)
        if num!=0:
            label=label+e+"$_{"+str(int(round(100*num,0)))+"}$"
    labels.append(label)
#for i in labels:
#    print(i)
plotter.linebatch_initial(points,labels,colors)
goal=setup.convert_points_to_new_projection('berny',setup.goal)
chosen_point=setup.convert_points_to_new_projection('berny',setup.chosen_point)
#plotter.linebatch_initial_chosen(points,labels,colors,chosen_point)
p=setup.values/np.sum(setup.values)
data=setup.convert_f_to_new_projection('berny',p,setup.omega)
centre=setup.convert_points_to_new_projection(
    'berny',setup.get_mean(f=p))
points=setup.convert_points_to_new_projection('berny',setup.points)
end_points=setup.get_end_points('berny')
labels=['Initial','Closest']
colors=['Blue','darkgreen']
#plotter.first_chosen(points,end_points,labels,colors)
#plotter.p_second_maxi_test(data,points,end_points,labels,colors,centre,goal)
#plotters=visualise_square()
#print('www',setup.points)
#d2=setup.make_heatmap_constrained()
#setup.points=setup.points[1:]
#setup.lines=setup.lines[1:]
#d1=setup.make_heatmap_constrained()
#plotters.test_heat(d0,d1,d2,setup.xlim,setup.ylim)
#plotter.p_second_max(data,points,end_points,labels,colors)
#plotter.p_second(data)
next_points=setup.choose_next_best_points_sphere(
    'mean',5,80,radius_reduction=20)
next_points_b=setup.convert_points_to_new_projection('berny',next_points)
closest_b=setup.convert_points_to_new_projection(
    'berny',setup.get_closest_point(next_points,index=False))
#plotter.second_batch(next_points_b)
setup.sample_next_batch(
    'provided',sigma,scale,delta,rietveld_closest=True,n_points=next_points)

points=setup.convert_points_to_new_projection('berny',setup.points)
end_points=setup.get_end_points('berny')
labels=['Initial','1st Closest','2nd Closest']
colors=['Blue','darkgreen','Lime']
#plotter.second_chosen(points,end_points,labels,colors)

#plot third p
p=setup.values/np.sum(setup.values)
data=setup.convert_f_to_new_projection('berny',p,setup.omega)
#plotter.p_third(data)

#choose third batch
next_points=setup.choose_next_best_points_sphere(
    'mean',5,80,set_radius=0.5)
setup.sample_next_batch(
    'provided',sigma,scale,delta,rietveld_closest=True,n_points=next_points)
next_points_b=setup.convert_points_to_new_projection('berny',next_points)
#plotter.third_batch(next_points_b)
chosen=setup.convert_points_to_new_projection('berny',setup.points[-1])
#plotter.third_closest(chosen)

#evaluate
#plotter.final(chosen,goal)
points=setup.convert_points_to_new_projection('berny',setup.points)
print(points[-1],'closest berny')
labels=['Initial','1st Closest','2nd Closest','3rd Closest']
colors=['Blue','darkgreen','Lime','Turquoise']
#plotter.final_testb(points,labels,colors,goal)
#plotter.final_testa(data,goal)

#print(next_points)
est_known=setup.get_estimated_known_composition(setup.points[-1])/100
closest_point=setup.convert_to_standard_basis(setup.points[-1])/100
goal=setup.convert_to_standard_basis(setup.goal)/100
print(goal,'goal')
print(closest_point,'closest')
print(est_known,'est known')
print('hjasdfasdf',np.sum(closest_point))
print('hjasdfasdf',np.sum(goal))
print('hjasdfasdf',np.sum(est_known))
#print(goal)
#print(closest_point)
#print(est_known)
a=np.stack([est_known,goal])
a=a.T
#print(a)
x=scipy.linalg.lstsq(a,closest_point)
k=x[0][0]*est_known
u=x[0][1]*goal
gt=u+k
print('Solution')
print(gt)
print(closest_point)
print(x[0])
phases=['Li','Zn','Si','S']
formula_known=setup.make_formula_for_molar_mass(phases,x[0][0]*est_known)
formula_goal=setup.make_formula_for_molar_mass(phases,x[0][1]*goal)
print(formula_known,'fk')
print(formula_goal,'fg')
mass_known=wt_convert.get_molar_mass(formula_known)[0]
mass_goal=wt_convert.get_molar_mass(formula_goal)[0]
print('mass known',mass_known)
print('mass goal',mass_goal)
goal_percent=mass_goal/(mass_goal+mass_known)
print(round(100*goal_percent,8),'%')

formula_known=setup.make_formula_for_molar_mass(phases,est_known)
formula_goal=setup.make_formula_for_molar_mass(phases,goal)
print('sumskg',np.sum(est_known),np.sum(goal))
print(formula_known,'fk')
print(formula_goal,'fg')
mass_known=wt_convert.get_molar_mass(formula_known)[0]
mass_goal=wt_convert.get_molar_mass(formula_goal)[0]
print('mass known',mass_known)
print('mass goal',mass_goal)
goal_percent=mass_goal*x[0][1]/(mass_goal*x[0][1]+mass_known*x[0][0])
known_percent=mass_known*x[0][0]/(mass_goal*x[0][1]+mass_known*x[0][0])
print(round(100*goal_percent,8),'%')
print(round(100*known_percent,8),'%')
print(known_percent/goal_percent)
print(goal_percent)
print('mass known = ',known_percent*0.5*10,'mg.')
'''
#code for getting resolution
'''
setup=all_information()
normal_a = np.array([1,2,4,-2])
normal_b = np.array([1,1,1,1])
normal_vectors=np.stack((normal_a,normal_b)) 
cube_size=100
delta_param=80
scale=1
delta=cube_size/delta_param
contained_point=np.array([2,1,1,4])*cube_size/8
setup.create_omega_constrained(
    normal_vectors,cube_size,contained_point,create_heatmap=True)
a=np.array([0,0])
b=np.array([0,1])
a_s=setup.convert_to_standard_basis(a)/cube_size
b_s=setup.convert_to_standard_basis(b)/cube_size
print(np.linalg.norm(a_s-b_s))
print(np.abs(a_s-b_s).max())
'''




'''
#code for getting results from setup
test=all_information()
overseer=Results()

num_trials=6000
output_file='../data/on_sphere_eval/compare_dist.csv'

setup_args={
    'Normal a':[1,2,4,-2],
    'Normal b':[1,1,1,1],
    'Cube size':100,
    'Delta param':80,
    'Scale':1,
    'Contained point':[2,1,1,4],
    'Batch size':5,
    'Sigma':0.1,
    'Rietveld closest':True,
    'Key param':'Compare'
}
setup_type=test.setup_one_batch_on_line

test_args={
    'Batch size':setup_args['Batch size'],
    'Centre':'mean',
    'Min angle':80,
    'Radius reduction':10,
    'Intercept':-66,
    'Slope':2.26,
    'Radius':3.5,
    'Custom radius':'yes',
}
#test_type=test.test_ball_batch_closest_variance
test_type=test.test_ball_batch_closest_variance_many
#test_type=None
#if isinstance(test_args['Radius reduction'],range):
#    test_type=test.test_ball_batch_closest_variance_many
#else:
#    test_type=test.test_ball_batch_closest_variance

result_descriptors=['Radius','Radius reduction','Regression']

overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)
    '''

#code for plotting
'''
output_file='../data/on_sphere_eval/.csv'
overseer=Results()
overseer.plot_test(output_file)
'''
#code for getting expected distance from Rietveld initialisation
'''
test=all_information()
overseer=Results()

num_trials=1000
output_file='../data/on_sphere_eval/e.csv'

setup_args={
    'normal_a':[1,2,4,-2],
    'normal_b':[1,1,1,1],
    'cube_size':100,
    'delta_param':80,
    'scale':1,
    'contained_point':[2,1,1,4],
    'batch_size':5,
    'formulas':['Li 4 Zn 0 Si 1 S 4',
                'Li 0 Zn 1 Si 0 S 1',
                'Li 2 Zn 0 Si 0 S 1'],
    'sampled_point':[24,10,9,40],
    'goal_point':[2,1,1,4],
    'weights':[0.51,0.40,0.09],
    'weight_error':[0.05],
}
setup_type=test.setup_one_batch_on_line_rietveld

test_args={
    'Radius reduction':10,
    'batch_size':setup_args['batch_size'],
    'centre':'mean',
    'min angle':80,
}
test_type=test.test_ball_batch_closest_variance

result_descriptors=['Closest distance', 'Max element % difference']
overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)
'''
#code for getting optimal radius Rietveld setup
'''
test=all_information()
overseer=Results()

num_trials=1000
output_file='../data/on_sphere_eval/radius_opt_Rieveld_1.csv'

setup_args={
    'Normal a':[1,2,4,-2],
    'Normal b':[1,1,1,1],
    'Cube size':100,
    'Delta param':80,
    'Scale':1,
    'Contained point':[2,1,1,4],
    'Batch size':5,
    'Formulas':['Li 4 Zn 0 Si 1 S 4',
                'Li 0 Zn 1 Si 0 S 1',
                'Li 2 Zn 0 Si 0 S 1'],
    'Sampled point':[24,10,9,40],
    'Goal point':[2,1,1,4],
    'Weights':[0.51,0.40,0.09],
    'Weight error':[0.05],
    'Radius':2.25,
    'Min angle':80,
    'Centre':'mean',
    'Ball batches':1,
}
setup_type=test.setup_rietveld_balls

test_args={
    'Radius':np.arange(0.1,2,0.01),
    'Batch size':setup_args['Batch size'],
    'Centre':'mean',
    'Min angle':80,
}
test_type=test.test_ball_batch_closest_variance_many

result_descriptors=['Closest distance', 'Radius']
overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)
    '''

#code for getting expected number of batches
'''
test=all_information()
overseer=Results()

num_trials=10
output_file='../data/on_sphere_eval/num_batches.csv'

setup_args={
    'Normal a':[1,2,4,-2],
    'Normal b':[1,1,1,1],
    'Cube size':100,
    'Delta param':80,
    'Scale':1,
    'Contained point':[2,1,1,4],
    'Batch size':5,
    'Sigma':0.1,
    'Rietveld closest':True
}
setup_type=test.setup_one_batch_on_line

test_args={
    'Radius reduction':10,
    'Batch size':setup_args['Batch size'],
    'Centre':'mean',
    'Min angle':80,
    'Sample type':'ball',
    'Condition type':'Closest distance',
    'Condition values':0.01,
    'Rietveld closest':True
}
test_type=test.test_num_batches_condition

result_descriptors=['Number of batches']
overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)
'''

#code for getting distance to mean vs variance
'''
test=all_information()
overseer=Results()

num_trials=6000
output_file='../data/on_sphere_eval/var_dist.csv'

setup_args={
    'Normal a':[1,2,4,-2],
    'Normal b':[1,1,1,1],
    'Cube size':100,
    'Delta param':80,
    'Scale':1,
    'Contained point':[2,1,1,4],
    'Batch size':5,
    'Sigma':0.1,
    'Rietveld closest':True
}
setup_type=test.setup_one_batch_on_line

test_args={}
test_type=test.test

result_descriptors=['Variance','Mean distance','Standard deviation']
overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)
'''
#code for getting results from setup
'''
test=all_information()
overseer=Results()

num_trials=1000
output_file='../data/on_sphere_eval/compare.csv'

setup_args={
    'Normal a':[1,3,-2,-2,-1],
    'Normal b':[1,1,1,1,1],
    'Cube size':30,
    'Delta param':80,
    'Scale':1,
    'Contained point':[2,1,1,1,1],
    'Batch size':5,
    'Sigma':0.1,
    'Rietveld closest':True,
}
setup_type=test.setup_one_batch_on_line

test_args={
    'Batch size':setup_args['Batch size'],
    'Centre':'mean',
    'Min angle':100,
    'Radius reduction':30,
    'Intercept':-26.4,
    'Slope':1.21,
    'Radius':0.88,
    'Custom radius':'yes',
}
#test_type=test.test_ball_batch_closest_variance
test_type=test.test_ball_batch_closest_variance_many
#test_type=None
#if isinstance(test_args['Radius reduction'],range):
#    test_type=test.test_ball_batch_closest_variance_many
#else:
#    test_type=test.test_ball_batch_closest_variance

result_descriptors=['Radius','Reduction','Regression']

overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)
    '''
#code for getting distance to mean vs variance
'''
test=all_information()
overseer=Results()

num_trials=1000
output_file='../data/on_sphere_eval/var_dist_5.csv'

setup_args={
    'Normal a':[1,3,-2,-2,-1],
    'Normal b':[1,1,1,1,1],
    'Cube size':30,
    'Delta param':80,
    'Scale':1,
    'Contained point':[2,1,1,1,1],
    'Batch size':5,
    'Sigma':0.1,
    'Rietveld closest':True
}
setup_type=test.setup_one_batch_on_line

test_args={}
test_type=test.test

result_descriptors=['Variance','Mean distance','Standard deviation']
overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)
'''
#code for random setup
'''
test=all_information()
overseer=Results()

num_trials=2000
output_file='../data/on_sphere_eval/var_mean_sig.csv'

setup_args={
    'Normal a':[1,2,4,-2],
    'Normal b':[1,1,1,1],
    'Cube size':100,
    'Delta param':80,
    'Scale':1,
    'Contained point':[2,1,1,4],
    'Sigma':0.3,
    'Max points':10

}
setup_type=test.setup_random

test_args={
}
test_type=test.test

result_descriptors=['Mean distance','Standard deviation']
overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)
    '''
#code to test n batches
test=all_information()
overseer=Results()

num_trials=600
output_file='../data/on_sphere_eval/n_batches_purity_6.csv'

test_args={
    'Multiple batches':True,
    'Closest distances':True,
    'Chebyshev distances':True,
    'Expected purities':True,
}

test_type=test.test_k
setup_args={
    'Normal a':[2,2,4,-2,-1.,-1],
    'Normal b':[1,1,1,1,1,1],
    'Cube size':20,
    'Delta param':80,
    'Scale':1,
    'Contained point':[1,1,1,4,1,1],
    'Sigma':0.3,
    'Rietveld closest':True,
    'Number of batches':10,
    'Slope':2.29,
    'Intercept':-65,
    #'Radius':5,
    'Centre':'mean',
    'Batch size':5,
    'Min angle':110,
    'Max':True,
}
setup_args.update(test_args)
setup_type=test.setup_n_batches_recording


result_descriptors=['Batch number','Closest distances','Chebyshev distances',
                    'Expected purities']
overseer.setup_test(
    setup_type,setup_args,test_type,test_args,result_descriptors,
    output_file=output_file,num_trials=num_trials)



