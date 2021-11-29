import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
path='../data/evaluation_opt/distances.txt'
df_all = pd.read_csv(path,sep=' ',header=0)
#Data Summary
print('--------')
for i in df_all.columns:
    print(i, df_all[i].unique())
print('--------')
for i in df_all.columns[:-1]:
    for j in df_all[i].unique():
        print(i,j,len(df_all[df_all[i]==j]))
print('--------')
df_all=df_all[df_all['fractional_cutoff']!=0.2]
#df=df_all[(df_all["number_points"] == 20) & (df_all['fractional_cutoff']==0.1)]
df=df_all[(df_all["number_points"] == 20)]
#df=df_all[(df_all['fractional_cutoff']==0.1)]
g = sns.relplot(
    data=df_all,
    x="number_targets", y="distance",
    col="number_points",
    hue="fractional_cutoff",style='fractional_cutoff',
    kind="line",
    palette="crest", linewidth=4, zorder=5,
    col_wrap=3, height=2, aspect=1.5,
)

for cutoff, ax in g.axes_dict.items():

    # Add the title as an annotation within the plot
    ax.text(.8, .85, cutoff, transform=ax.transAxes, fontweight="bold")

    # Plot every year's time series in the background
    '''
    sns.lineplot(
        data=dfa, x="number_targets", y="distance", units="fractional_cutoff",
        estimator=None, color=".7", linewidth=1, ax=ax,
    )
    '''
plt.show()


