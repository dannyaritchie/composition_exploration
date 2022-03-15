import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns

#load data
path='../data/evaluation_opt/distancesa.txt'
df = pd.read_csv(path,sep=' ',header=0)
#Data Summary
print('--------')
for i in df.columns:
    print(i, df[i].unique())
print('--------')
for i in df.columns[:-1]:
    for j in df[i].unique():
        print(i,j,len(df[df[i]==j]))
print('--------')

dim = 3
#number_points [10 20]
#number_targets [10 15 20 25]
theta_range = 20
theta_distribution = 'uni'
angular_equivalence = 20
increment = 1
fractional_cutoff = 0.1
batch_size = 3
k = 3.14159265

df=df[(df['fractional_cutoff']==0.1)]
df=df[(df['dim']==3)]
df=df[(df['increment']==1)]
df=df[(df['theta_range']==20)]
df=df[(df['theta_distribution']=='uniform')]
df=df[(df['angular_equivalence']==20)]
df=df[(df['batch_size']==3)]
df=df[(df['k']==round(np.pi,15))]

# Create figure
fig = go.Figure()
# Add traces, one for each slider step
for number_points in sorted(df['number_points'].unique()):
    means=df[df['number_points']==number_points].groupby('number_targets')['distance'].mean()
    error_mean=df[df['number_points']==number_points].groupby('number_targets')['distance'].sem()
    count=df[df['number_points']==number_points].groupby('number_targets')['distance'].count()
    fig.add_trace(
        go.Scatter(
            meta=count,
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="Numbbbber of points = " + str(number_points),
            x=means.index,
            error_y=go.scatter.ErrorY(array=error_mean),
            hovertemplate = 'Number of repeats: %{meta}<extra></extra>',
            y=means ))
# Make 10th trace visible
fig.data[0].visible = True
# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        label=str(sorted(df['number_points'].unique())[i]),
        args=[{"visible": [False] * len(fig.data)},
              ],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Number of points: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    title="k=pi,dim=3,a=0.1,t_rng=20,t_dist=uni,inc=1,ang_equ=20,b=3",
    xaxis_title="Number of trial targets",
    yaxis_title="Mean distance to true target",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
fig.update_xaxes(tickmode='array',
                  tickvals=[10,15,20,25])

fig.show()
