from algorithm import *
import pandas as pd
import os.path
import seaborn as sns
from scipy.stats import linregress

class Results:

    phase_field_dict={'A':[1,2,-2,-2]}
    angle_parameter_dict={'A':{'k':np.pi}}
    gaussian_parameter_dict={'A':{'scale':1,'delta':1},
                             'B':{'scale':2,'delta':1},
                             'C':{'scale':0.5,'delta':1},
                             'D':{'scale':1,'delta':0.5},
                             'E':{'scale':1,'delta':2},
                            }



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
        dist=0.05 #simulated distance of avg known composition to sample
        self.sigma=np.diag(np.array([0.0188,0.0188])/dist)
        self.score_method='d_g_mu'
        self.p_method='angle_product'
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
            for parameter in ['A','B']:
                self.n=n
                self.parameter_key=parameter
                self.save_results(filename,n_trials=m)

    def plot_mean_vs(self,vs,selection_dic,filename,save_path=None):
        df=pd.read_csv(self.directory+filename)
        #for column in selection_dic:
        #    df=df[df[column]==selection_dic[column]]
        print(len(df))
        sns.catplot(x=vs,y='Score',data=df,kind='box')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def get_score_vs_add_points(self,filename):
        normal_a = np.array([1,2,-2,-2])
        normal_b = np.array([1,1,1,1])
        self.normal_vectors=np.stack((normal_a,normal_b)) 
        self.cube_size=100
        self.contained_point=np.array([2,1,1,1])*self.cube_size/5
        dist=0.05 #simulated distance of avg known composition to sample
        self.sigma=np.diag(np.array([0.0188,0.0188])/dist)
        test=all_information()

        number_samples=5
        scale=1
        delta=1
        power=1
        number_repeats=1
        scores=np.empty((number_repeats,number_samples+1))
        for i in range(number_repeats):
            print(i)
            scores[i]=test.sample_points_test(
                number_samples,self.normal_vectors,self.contained_point,
                self.cube_size,self.sigma,scale,delta,power,plot_process=True)
        np.savetxt(self.directory+filename,scores)

    def plot_score_vs_add_point(self,filename):
        fig,ax=plt.subplots(1,2)
        scores=np.loadtxt(self.directory+filename)
        print(scores.shape)
        meana = np.mean(scores,axis=0)
        stda=np.std(scores,axis=0)
        ax[1].errorbar(
            x=range(5),y=meana,yerr=stda,ecolor='red',elinewidth=1,ls='',
            marker='x',)
        filename='s1_d1_p2_a.txt'
        scores=np.loadtxt(self.directory+filename)
        meanb = np.mean(scores,axis=0)
        stdb=np.std(scores,axis=0)
        ax[0].errorbar(
            x=range(5),y=meanb,yerr=stdb,ecolor='red',elinewidth=1,ls='',
            marker='x',)
        plt.show()
        
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

    def setup_test(
            self,setup_type,setup_args,test_type,test_args,result_descriptors,
            output_file="",key_param=None,plot_type=None,plot_args=None,
            num_trials=2):

        test=all_information()
        results = np.empty((0,len(result_descriptors)))

        for i in range(num_trials):
            print(i)
            setup_type(setup_args)
            result=test_type(test_args,result_descriptors)
            if result is not None:
                results = np.append(results,result,axis=0)

        df = pd.DataFrame(data=results,columns=result_descriptors)
        for setup_type in setup_args.keys():
            if not setup_type in result_descriptors:
                if np.isscalar(setup_args[setup_type]):
                    df[setup_type]=setup_args[setup_type]
                else:
                    df[setup_type]=str(setup_args[setup_type])
        for test_type in test_args.keys():
            if not test_type in result_descriptors:
                print(test_type)
                df[test_type]=test_args[test_type]

        if output_file!="":
            df.to_csv(
                output_file,mode='a',header=not os.path.exists(output_file),
                index=False)

    def plot_test(self,path,x,y):
        df=pd.read_csv(path)
        #sns.jointplot(data=df,x=x,y=y,kind='kde')
        sns.histplot(data=df,x=x)
        #sns.lineplot(x='Standard deviation',y='Mean distance',data=df)
        plt.show()

    def plot_line(self,path,x,y):
        df=pd.read_csv(path)
        #df=df[df[x]<=2]
        sns.lineplot(data=df,x=x,y=y)#.set_xlabel('K')#.set(
            #title="Closest distance vs ball radius for n=6000 after line\n"
            #+"batch setup with rietveld closest")
        plt.show()

    def plot_lines_melt(self,path,x,y1,y2,title=""):
        test=all_information()
        normal_a = np.array([1,2,-2,-2])
        normal_b = np.array([1,1,1,1])
        normal_vectors=np.stack((normal_a,normal_b)) 
        cube_size=100
        contained_point=np.array([1,1,1,2])*cube_size/5
        sigma=np.diag(np.array([0.1,0.1]))
        test.setup(normal_vectors,contained_point,cube_size,sigma)
        a=np.array([0,0])
        b=np.array([0,1])
        a_s=test.convert_to_standard_basis(a)/cube_size
        b_s=test.convert_to_standard_basis(b)/cube_size
        res=np.abs(a_s-b_s).max()/2

        df=pd.read_csv(path)
        df=df[[x,y1,y2]]
        df=df.melt(x,var_name='Score type',value_name='Score')
        ax=sns.lineplot(data=df,x=x,y='Score',hue='Score type')#.set_xlabel('K')#.set(
            #title="Closest distance vs ball radius for n=6000 after line\n"
            #+"batch setup with rietveld closest")
        ax.axhline(
            y=res,color='Red',
            label='Resolution:'+str(round(res,4)))
        plt.legend()
        plt.title(title)
        plt.show()

    def plot_line_melt(self,path,x,y,title=""):
        test=all_information()
        normal_a = np.array([1,2,-2,-2])
        normal_b = np.array([1,1,1,1])
        normal_vectors=np.stack((normal_a,normal_b)) 
        cube_size=100
        contained_point=np.array([1,1,1,2])*cube_size/5
        sigma=np.diag(np.array([0.1,0.1]))
        test.setup(normal_vectors,contained_point,cube_size,sigma)
        goal=[cube_size/2,cube_size/2]
        test.goal=goal
        point=[cube_size/2,cube_size/2]
        point[0]-=1
        point[1]+=1
        points=np.array([point])
        print(points)
        
        res=test.get_expected_purity(points,cheat=True)
        print(res)
        #a=np.array([0,0])
        #b=np.array([0,1])
        #a_s=test.convert_to_standard_basis(a)/cube_size
        #b_s=test.convert_to_standard_basis(b)/cube_size
        #res=np.abs(a_s-b_s).max()/2

        df=pd.read_csv(path)
        print(df.columns)
        print(len(df))
        #df=df.tail(10000)
        print(df['Expected purities'])
        df=df[[x,y]]
        df=df.melt(x,var_name='Score type',value_name='Score')
        ax=sns.lineplot(data=df,x=x,y='Score',hue='Score type')#.set_xlabel('K')#.set(
            #title="Closest distance vs ball radius for n=6000 after line\n"
            #+"batch setup with rietveld closest")
        #ax.axhline(
            #y=res,color='Red',
            #label='Resolution:'+str(round(res,4)))
        ax.set_ylabel('Expected purity / %')
        ax.get_legend().remove()
        plt.title(title)
        plt.show()

    def plot_hists(self,patha,pathb,x):
        dfa=pd.read_csv(patha)
        dfb=pd.read_csv(pathb)
        df=dfa[['Closest distance','Key param']]
        df=df.append(dfb[['Closest distance','Key param']])
        print(df['Key param'].unique())
        g=sns.FacetGrid(df,row='Key param')
        g.map(sns.histplot,'Closest distance')
        plt.show()
        '''
        plt.suptitle('Distance from closest sampled point to goal\n'
                     +'after 1 line sample and 1 sphere sample with\n'
                     +'linear radius fit,b=5,s=0.1')
        axs[0].axvline(
            x=dfa[x].mean(),color='Red',
            label='LFRB Mean:'+str(round(dfa[x].mean(),3)))
        axs[1].axvline(
            x=dfb[x].mean(),color='Red',
            label='VRRB Mean:'+str(round(dfb[x].mean(),3)))
        '''

    def plot_hist(self,path,x,title=None):
        df=pd.read_csv(path)
        dfa=df[df['Key param']=='VRRB']
        dfb=df[df['Key param']=='LFRB']
        #sns.jointplot(data=df,x=x,y=y,kind='kde')
        fig,axs=plt.subplots(2)
        sns.histplot(data=dfa,x=x,ax=axs[0])
        sns.histplot(data=dfb,x=x,ax=axs[1])
        axs[0].set_title("VRRB")
        axs[1].set_title("LFRB")
        #sns.lineplot(x='Standard deviation',y='Mean distance',data=df)
        if title is not None:
            plt.suptitle('Distance from closest sampled point to goal\n'
                         +'after 1 line sample and 1 sphere sample with\n'
                         +'linear radius fit,b=5,s=0.1')
        axs[0].axvline(
            x=dfa[x].mean(),color='Red',
            label='Mean:'+str(round(dfa[x].mean(),3)))
        axs[1].axvline(
            x=dfb[x].mean(),color='Red',
            label='Mean:'+str(round(dfb[x].mean(),3)))
        plt.legend()
        plt.show()

    def plot_regression(self,a,b,path):
        df=pd.read_csv(path)
#        df=df[df['Mean distance']>0]
#        df=df[df['Standard deviation']>30]
        #df['STD']=np.sqrt(df['Variance from max'])

        regress=linregress(df[a],df[b])
        print(regress.slope)
        print(regress.intercept)
        print(regress.rvalue)
        rel=sns.jointplot(data=df,x=a,y=b,kind='reg',scatter_kws={"s":0.3})
        rel.fig.suptitle(
            'Linear fit: slope='+str(round(regress.slope,4))+", intercept="
            +str(round(regress.intercept,4))+"\nrvalue="
            +str(round(regress.rvalue,4)))
        plt.show()

    def write_stats(self,path,stats,x,print=False,write=None):
        '''
        df=pd.read_csv(path)
        stats=''
        for s in stats:
            stats+=s+": "
            if s=='Mean':
                stats+=str(round(s,4))
            stats+=", "
        if write is not None:
        '''

                




