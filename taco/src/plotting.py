import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

current_palette = sb.color_palette()
cp = [list(c)+[0.5] for c in current_palette]
cp[1] =cp[3]
cp[3] =cp[0]
cp[0] = [168/255.,194/255.,251/255.]
cp = [tuple(c) for c in cp]
cut = cp[:4]


sb.set_style("whitegrid")
plt.rcParams['figure.figsize'] = 200, 200
tips = sb.load_dataset("tips")
def factorplot(labels,data,name,):
    sb.set_style("whitegrid",{'grid.linewidth': 3.,
                            'axes.edgecolor':'0.0','grid.edgecolor':'0.0'})
    g = sb.factorplot(labels[0],labels[1],labels[2],kind='bar', data=data,legend_out=False,palette=cp,
                     hue_order = ['Oracle','TACO',"Naive (MLP)","Naive (GRU)"] )

    sb.set_style("whitegrid")

    g.set(ylim = (0,1))
    g.set(xlim=(-0.45, 3.01))
    g.fig.set_size_inches(15,8)
    plt.legend(loc='best')
    g.savefig(name+'.pdf')

