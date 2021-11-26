import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
path='evaluation_opt/distances.txt'
df = pd.read_csv(path,sep=' ',header=0)
for i in df.columns:
    print(i)
df=df[(df['number_points']==15) | (df['number_points']==25)]
dfa=df[df['number_points']==15]
dfb=df[df['number_points']==25]
dfaa=dfa[dfa['fractional_cutoff']==0.1]
dfab=dfa[dfa['fractional_cutoff']==0.2]
dfba=dfb[dfb['fractional_cutoff']==0.1]
dfbb=dfb[dfb['fractional_cutoff']==0.2]

g = sns.relplot(
    data=df,
    x="number_targets", y="distance", col="number_points",
    hue="fractional_cutoff", size='fractional_cutoff',
    style='fractional_cutoff',
    kind="line", palette="crest", linewidth=4, zorder=5,
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


