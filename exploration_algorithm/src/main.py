import ternary
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.ma as ma
from algorithm import *
from pathlib import Path

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
dim = 3
theta_distribution='uniform'
theta_range=20
increment=1
angular_equivalence=20
name='evaluation_opt/3d/'
batch_size=3
k=np.pi
p = Path('evaluation_opt/distances.txt')
if not p.is_file():
    print('hi')
    f = p.open(mode = 'w')
    f.write('dim' + ' ' + 'numer_points' + ' ' + 'number_targets' + ' ' +
            'theta_range' + ' ' + 'theta_distribution' + ' ' +
            'angular_equivalence' + ' ' + 'increment' + ' ' +
            'fractional_cutoff' + ' ' + 'batch_size' + ' ' + 'k' + ' ' +
            'distance' + '\n')
    f.close()
f=p.open(mode = 'a')
for i in range(1):
    for a in [0.1]:
        namea=name + str(a)
        for m in [100]:
            print(i)
            nameam = namea + '_' + str(m)
            for n in [40]:
                nameamn = nameam + '_' + str(n) + '.eps'
                #nameamn=''
                setup=all_information()
                distance=setup.evaluation(dim,n,m,theta_range,theta_distribution,angular_equivalence,increment,a,batch_size,k,nameamn,plot_process=True)
                f.write(str(dim) + ' ' + str(n) + ' ' + str(m) + ' ' +
                        str(theta_range) + ' ' + theta_distribution + ' ' +
                        str(angular_equivalence) + ' ' + str(increment) + ' ' +
                        str(a) + ' ' + str(batch_size) + ' ' + str(k) + ' ' +
                        str(distance) + '\n')
f.close()
 

