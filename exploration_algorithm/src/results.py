from algorithm import *
import pandas as pd
import os.path
import seaborn as sns

class Results:

    phase_field_dict={'A':[1,2,-2,-2]}
    angle_parameter_dict={'A':{'k':np.pi}}
    gaussian_parameter_dict={'A':{'scale':1,'delta':1},
                             'B':{'scale':2,'delta':1},
                             'C':{'scale':0.5,'delta':1},
                             'D':{'scale':1,'delta':0.5},
                             'E':{'scale':1,'delta':2},
                            }


    def __init__(self,save_directory):
        self.directory=save_directory

    def score_at_n_points(
        self,score_method,n,p_method,sigma,cube_size,normal_vectors,
        contained_point,scale=1,k=np.pi,delta=1):
        test=all_information()
        test.setup(normal_vectors,contained_point,cube_size,sigma)
        test.random_initialise(n)
        if p_method=='angle_product':
            test.prod_theta_score_n_constrained(k=k,return_type='normal')
        elif p_method=='guassian':
            test.make_p_gaussian(sigma,scale,delta)
        else:
            print('Error: Unknown p method')

        return test.get_score(score_method)

    def test(self):
        normal_a = np.array([1,2,-2,-2])
        normal_b = np.array([1,1,1,1])
        normal_vectors=np.stack((normal_a,normal_b)) 
        cube_size=100
        contained_point=np.array([1,1,1,2])*cube_size/5
        dist=0.5 #simulated distance of avg known composition to sample
        sigma=np.diag(np.array([0.0188,0.0188])/dist)
        score=self.score_at_n_points(
            'd_g_mu',3,'angle_product',sigma,cube_size,normal_vectors,
            contained_point)
        print(score)

    def set_default_columns(self):
        self.columns=['Score method','P method','Number of points','std',
                 'cube_size','Phase field key','Parameter key','Score']
        normal_a = np.array([1,2,-2,-2])
        normal_b = np.array([1,1,1,1])
        self.normal_vectors=np.stack((normal_a,normal_b)) 
        self.cube_size=100
        self.contained_point=np.array([2,1,1,1])*self.cube_size/5
        dist=0.5 #simulated distance of avg known composition to sample
        self.sigma=np.diag(np.array([0.0188,0.0188])/dist)
        self.score_method='d_g_mu'
        self.p_method='guassian'
        self.n=3
        self.parameter_key='A'

    def save_results(self,filename,save=True,n_trials=10):
        data=[[None]]*n_trials
        for i in range(n_trials):
            print('Trial num: ',i)
            tdict=Results.gaussian_parameter_dict
            scale=tdict[self.parameter_key]['scale']
            delta=tdict[self.parameter_key]['delta']
            print('scale: ',scale)
            score=self.score_at_n_points(
                self.score_method,self.n,self.p_method,self.sigma,
                self.cube_size,self.normal_vectors,self.contained_point,
                scale=scale,delta=delta)
            parameters=[self.score_method,self.p_method,self.n,
                        self.sigma[0][0],self.cube_size,'A',self.parameter_key,
                        score]
            data[i]=parameters
        df=pd.DataFrame(data,columns=self.columns)
        if save:
            path=self.directory+filename
            df.to_csv(
                path,mode='a',header=not os.path.exists(path),index=False)

    def two_d_results(self,filename,m=10):
        self.set_default_columns()
        for n in [4]:
            for parameter in ['D','E']:
                self.n=n
                self.parameter_key=parameter
                self.save_results(filename,n_trials=m)

    def plot_mean_vs(self,vs,selection_dic,filename,save_path=None):
        df=pd.read_csv(self.directory+filename)
        for column in selection_dic:
            df=df[df[column]==selection_dic[column]]
        sns.catplot(x=vs,y='Score',data=df,kind='box')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def get_score_vs_add_points(self):
        normal_a = np.array([1,2,-2,-2])
        normal_b = np.array([1,1,1,1])
        self.normal_vectors=np.stack((normal_a,normal_b)) 
        self.cube_size=100
        self.contained_point=np.array([2,1,1,1])*self.cube_size/5
        dist=0.5 #simulated distance of avg known composition to sample
        self.sigma=np.diag(np.array([0.0188,0.0188])/dist)
        test=all_information()

        filename='s1_d1_p2_a.txt'
        number_samples=10
        scale=1
        delta=1
        power=2
        number_repeats=1000
        scores=np.empty((number_repeats,number_samples+1))
        for i in range(number_repeats):
            print(i)
            scores[i]=test.sample_points_test(
                number_samples,self.normal_vectors,self.contained_point,
                self.cube_size,self.sigma,scale,delta,power)
        np.savetxt(self.directory+filename,scores)

    def plot_score_vs_add_point(self):
        fig,ax=plt.subplots(1,2)
        filename='s1_d1_p1_a.txt'
        scores=np.loadtxt(self.directory+filename)
        meana = np.mean(scores,axis=0)
        stda=np.std(scores,axis=0)
        ax[1].errorbar(
            x=range(11),y=meana,yerr=stda,ecolor='red',elinewidth=1,ls='',
            marker='x',)
        filename='s1_d1_p2_a.txt'
        scores=np.loadtxt(self.directory+filename)
        meanb = np.mean(scores,axis=0)
        stdb=np.std(scores,axis=0)
        ax[0].errorbar(
            x=range(11),y=meanb,yerr=stdb,ecolor='red',elinewidth=1,ls='',
            marker='x',)
        plt.show()
        

        





    
