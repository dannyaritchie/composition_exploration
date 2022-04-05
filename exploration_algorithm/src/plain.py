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

overseer=Results()

output_filea='../data/on_sphere_eval/radius_opt_5.csv'
output_fileb='../data/on_sphere_eval/reduction.csv'
output_file='../data/on_sphere_eval/radius_opt_Rieveld_1.csv'
'''
df=pd.read_csv(output_file)
dfa=pd.DataFrame()
dfb=pd.DataFrame()
dfc=pd.DataFrame()
dfa['Closest distance']=df['Radius']
dfa['Radius type']='Set radius'
print(dfa.columns)
print(len(dfa))
#dfb['Closest distance']=df['Radius reduction']
#dfb['Radius type']='Radius reduction'
dfc['Closest distance']=df['Regression']
dfc['Radius type']='Regression'
df=dfa.append(dfc)
#df=df.append(dfc)
print(df.columns)
sns.pointplot(
    data=df,x='Radius type',y='Closest distance',linestyles="",capsize=0.1,
    markers='o',errwidth=1,linewidth=0.5,s=5)
plt.show()
'''

#g=sns.FacetGrid(df,row='Radius type')
#g.map(sns.histplot,'Closest distance',bins=100)

#overseer.plot_regression('Standard deviation','Mean distance',output_file)
#overseer.plot_hists(output_filea,output_fileb,'Closest distance')
overseer.plot_line(output_file,'Radius','Closest distance')
#overseer.plot_line_melt(output_file,'Batch number','Closest distance',
#                       'Max individual distance',title='With max')
#code to alter file
'''
df=pd.read_csv(output_file)
df['Key param']='LFRB'
df.to_csv(output_file,index=False)
'''
#overseer.plot_regression('Standard deviation','Mean distance',output_filea)
