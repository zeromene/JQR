"""Re-render the seven empirical figures at their actual manuscript width."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.text import Text

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
D=json.loads((HERE/'figure_data.json').read_text(encoding='utf-8'))
B=json.loads((HERE/'budget_sweep_summary.json').read_text(encoding='utf-8'))
OUT=ROOT/'english/figures'
PRE=ROOT/'figure_previews';PRE.mkdir(exist_ok=True)
W=468/72.27
TEAL='#087F83';NAVY='#355B88';GOLD='#B37C25';RED='#B34D55';GRAY='#788690'
INK='#202B34';LIGHT='#E4EAED'
STYLES=[(NAVY,'o'),(TEAL,'s'),(GOLD,'^'),(RED,'D')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
 'axes.labelsize':10.5,'axes.titlesize':11,'xtick.labelsize':10,'ytick.labelsize':10,
 'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white',
 'axes.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.7,'mathtext.fontset':'dejavusans'})
REPORT={}

def save(fig,name,title):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();box=fig.bbox
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    outside=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        if min(b.x0-box.x0,b.y0-box.y0,box.x1-b.x1,box.y1-b.y1)<-1:
            outside.append(t.get_text())
    REPORT[name]={'width_inches':W,'height_inches':float(fig.get_size_inches()[1]),
                  'minimum_text_points':min(t.get_fontsize() for t in texts),
                  'out_of_canvas_text':outside}
    fig.savefig(OUT/(name+'.pdf'),metadata={'Title':title,'Creator':'Matplotlib'})
    fig.savefig(PRE/(name+'.png'),dpi=170)
    shutil.copy2(OUT/(name+'.pdf'),ROOT/'chinese/figures'/(name+'.pdf'))
    plt.close(fig)

def hops():
    fig=plt.figure(figsize=(W,4.65))
    cmap=LinearSegmentedColormap.from_list('quality',['#FAFAFA','#DCDCDC','#A5A5A5','#535353'])
    for metric,left,title in [('f1',.285,'F1'),('em',.65,'Exact match')]:
        ax=fig.add_axes([left,.17,.325,.67])
        vals=np.array([[c[metric] for c in row['cells']] for row in D['hops']])
        im=ax.pcolormesh(np.arange(4)-.5,np.arange(11)-.5,vals,cmap=cmap,vmin=0,vmax=70,shading='flat')
        ax.set(xlim=(-.5,2.5),ylim=(9.5,-.5))
        for i,row in enumerate(D['hops']):
            ours=row['method']=='quote_anchor'
            if ours:ax.add_patch(Rectangle((-.5,i-.5),3,1,fc=TEAL,ec='none',zorder=2))
            for j in range(3):
                ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',fontsize=11,
                        fontweight='bold' if ours else 'normal',color='white' if ours or vals[i,j]>46 else INK)
        ax.set_xticks(range(3),['2-hop\n249','3-hop\n151','4-hop\n80']);ax.xaxis.tick_top()
        ax.set_yticks(range(10),[r['label'].replace('Reranker + packing','Reranker packing') for r in D['hops']] if metric=='f1' else ['']*10)
        ax.tick_params(which='both',length=0,pad=5,labelsize=10.5)
        for tick in ax.get_yticklabels():
            if tick.get_text()=='JQR':tick.set_color(TEAL);tick.set_weight('bold')
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True)
        ax.grid(which='minor',color='white',lw=1.1)
        for spine in ax.spines.values():spine.set_visible(False)
        fig.text(left+.1625,.97,title,ha='center',va='top',fontsize=11.5,fontweight='bold')
    cb=fig.colorbar(im,cax=fig.add_axes([.35,.075,.56,.018]),orientation='horizontal',ticks=[0,20,40,60,70])
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=10,length=2,pad=3);cb.solids.set_rasterized(False)
    save(fig,'fig2_hop_profiles','Answer quality by reasoning depth')

def budgets():
    fig=plt.figure(figsize=(W,5.3))
    xs=[128,256,384,512]
    datasets=[('hotpot','HotpotQA'),('musique','MuSiQue')]
    methods=[('quote_anchor','JQR',TEAL,'o'),('strong_rerank','Reranker packing','#60748A','s')]
    for i,(ds,title) in enumerate(datasets):
        for j,reader in enumerate(['Qwen3.5-9B','Phi-4-mini-instruct']):
            ax=fig.add_axes([.10+j*.50,[.625,.265][i],.38,.24])
            for method,label,color,marker in methods:
                ys=[next(g['f1'] for g in B['groups'] if (g['dataset'],g['reader'],g['budget'],g['method'])==(ds,reader,x,method)) for x in xs]
                ax.plot(xs,ys,marker=marker,color=color,label=label,lw=1.6,ms=4.5)
            ax.set_title(title+' / '+reader,fontsize=10.2,pad=8)
            ax.set(xlim=(108,532),ylim=(20,90),yticks=[20,40,60,80],xticks=xs)
            ax.grid(axis='y',color=LIGHT,lw=.65)
            if j==0:ax.set_ylabel('Answer F1')
            if i==1:ax.set_xlabel('Budget (Qwen tokens)',labelpad=5)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=1.6,label=l) for _,l,c,m in methods],
               loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,frameon=False,fontsize=10)
    cmap=LinearSegmentedColormap.from_list('admit',['#F2F7F6','#75B4B1'])
    for j,(ds,title) in enumerate(datasets):
        ax=fig.add_axes([.10+j*.50,.055,.38,.045]);vals=[next(r['admission_percent'] for r in B['preparation']['summaries'] if (r['dataset'],r['budget'])==(ds,x)) for x in xs]
        ax.pcolormesh(np.array(xs+[640])-64,[0,1],[vals],cmap=cmap,vmin=0,vmax=100,edgecolors='white',linewidth=1.6)
        for x,v in zip(xs,vals):ax.text(x,.5,f'{v:.1f}%',ha='center',va='center',fontsize=10,color='#163A3A')
        ax.set(xlim=(64,576),ylim=(0,1),yticks=[],xticks=xs)
        ax.set_title(title+' admission',fontsize=10.5,pad=6);ax.tick_params(length=0,labelsize=10)
        for spine in ax.spines.values():spine.set_visible(False)
    save(fig,'budget_quality','Budget sweep: answer quality and quotation admission')

def raincloud():
    fig=plt.figure(figsize=(W,4.45));ax=fig.add_axes([.24,.14,.735,.82])
    keys=['quote_anchor','qwen_rerank_sentence','brief_pro_auto_cap','exit_budgeted','compact_budgeted','longrefiner_budgeted_repair_v1']
    names=['JQR','Reranker packing','BRIEF-Pro','EXIT','CompAct','LongRefiner']
    colors=[TEAL,GRAY,NAVY,'#8C6EA3',GOLD,'#4C8795'];grid=np.linspace(0,512,700);rng=np.random.default_rng(30)
    for i,(key,color) in enumerate(zip(keys,colors)):
        v=np.array([r['evidence_tokens'] for r in D['public_records'][key]]);y=5-i
        den=sum(np.exp(-.5*((grid[:,None]-u[None,:])/12)**2).mean(1) for u in [v,-v,1024-v])/(12*np.sqrt(2*np.pi))
        ax.fill_between(grid,y+.025,y+.025+.43*den/den.max(),color=color,alpha=.45,linewidth=.7,edgecolor=color)
        ax.scatter(v,y-.14+rng.uniform(-.055,.055,len(v)),s=3,color=color,alpha=.30,linewidth=0,rasterized=False)
        q10,q25,q50,q75,q90=np.quantile(v,[.1,.25,.5,.75,.9])
        ax.plot([q10,q90],[y-.01,y-.01],color=INK,lw=1)
        ax.add_patch(Rectangle((q25,y-.046),q75-q25,.072,fc='white',ec=INK,lw=.8,zorder=5))
        ax.plot([q50,q50],[y-.046,y+.026],color=INK,lw=1.4,zorder=6)
        if key in ['quote_anchor','brief_pro_auto_cap']:
            ax.annotate(f'Median {q50:.1f}',xy=(q50,y+.04),xytext=(q50+55,y+.34),fontsize=10.5,
                        ha='left',va='center',color=INK,bbox=dict(facecolor='white',edgecolor='none',pad=1.0,alpha=.92),
                        arrowprops=dict(arrowstyle='-',color=INK,lw=.7),zorder=7)
    ax.axvline(256,color=LIGHT,ls='--',lw=.8,zorder=0);ax.axvline(512,color=GRAY,lw=.6)
    ax.set(xlim=(-5,521),ylim=(-.42,5.63),yticks=range(6),yticklabels=names[::-1],xticks=[0,128,256,384,512],xlabel='Actual evidence tokens (Qwen)')
    ax.tick_params(axis='y',length=0,pad=8,labelsize=10.5);ax.spines['left'].set_visible(False)
    for t in ax.get_yticklabels():
        if t.get_text()=='JQR':t.set_color(TEAL);t.set_weight('bold')
    save(fig,'fig6_length_distributions','Question-level evidence length distributions')

def ribbon(ax,x0,x1,y0,y1,h,color,alpha):
    dx=(x1-x0)*.46
    v=[(x0,y0),(x0+dx,y0),(x1-dx,y1),(x1,y1),(x1,y1+h),(x1-dx,y1+h),(x0+dx,y0+h),(x0,y0+h),(x0,y0)]
    ax.add_patch(PathPatch(MPath(v,[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]),fc=color,ec='none',alpha=alpha))

def flows():
    fig=plt.figure(figsize=(W,4.3))
    for k,flow in enumerate(D['flows']):
        ax=fig.add_axes([.035+k*.495,.24,.46,.61]);ax.set(xlim=(0,1),ylim=(0,1.14));ax.axis('off')
        mat=np.array(flow['matrix']);left=mat.sum(1);right=mat.sum(0);scale=.91/480;gap=.04
        lbase=np.r_[0,np.cumsum(left[:-1])*scale]+np.arange(3)*gap
        rbase=np.r_[0,np.cumsum(right[:-1])*scale]+np.arange(3)*gap;lo=lbase.copy();ro=rbase.copy()
        for i in range(3):
            for j in range(3):
                h=mat[i,j]*scale
                if h:ribbon(ax,.29,.70,lo[i],ro[j],h,TEAL if j>i else RED if j<i else '#BDCAD0',.70 if i!=j else .45)
                lo[i]+=h;ro[j]+=h
        for totals,base,x,tx,ha in [(left,lbase,.24,.225,'right'),(right,rbase,.70,.765,'left')]:
            for i,n in enumerate(totals):
                ax.add_patch(Rectangle((x,base[i]),.05,n*scale,fc=[GRAY,GOLD,NAVY][i],ec='white',lw=.6))
                ax.text(tx,base[i]+n*scale/2,['F1 = 0','0 < F1 < 1','F1 = 1'][i]+f'\n{n}',ha=ha,va='center',fontsize=9.5,linespacing=1.2)
        ax.text(.265,1.095,flow['label'].replace('Reranker + packing','Reranker packing'),ha='center',fontsize=10.5,fontweight='bold')
        ax.text(.725,1.095,'JQR',ha='center',fontsize=10.5,fontweight='bold',color=TEAL)
        x=.265+k*.495
        fig.text(x,.145,f"Recovered {flow['rescued']}/{flow['at_risk']} ({100*flow['rescued']/flow['at_risk']:.1f}%)",ha='center',fontsize=10)
        fig.text(x,.085,f"Preserved {flow['retained']}/{flow['initial_full']} ({100*flow['retained']/flow['initial_full']:.1f}%)",ha='center',fontsize=10)
    fig.legend(handles=[Line2D([],[],color=c,lw=5,label=t) for c,t in [(TEAL,'Higher F1 bin'),('#BDCAD0','Same bin'),(RED,'Lower F1 bin')]],
               loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False,fontsize=10,columnspacing=1.4)
    save(fig,'fig3_answer_transitions','Paired full, partial and zero-F1 transitions')

def mosaics():
    fig=plt.figure(figsize=(W,5.6));cmap=LinearSegmentedColormap.from_list('delta',[RED,'#EEF0F1',TEAL])
    for k,cell in enumerate(D['mosaics']):
        col=k%2;row=k//2;ax=fig.add_axes([.09+col*.47,[.625,.18][row],.35,.31])
        vals=np.array([x['delta'] for x in cell['rows']]).reshape(16,16)
        im=ax.pcolormesh(np.arange(17)-.5,np.arange(17)-.5,vals,cmap=cmap,norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=100),shading='flat',rasterized=False)
        ax.set(xlim=(-.5,15.5),ylim=(15.5,-.5),aspect='equal')
        ax.set_xticks(np.arange(-.5,16,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True)
        ax.grid(which='minor',color='white',lw=.45);ax.tick_params(which='both',length=0,labelbottom=False,labelleft=False)
        for spine in ax.spines.values():spine.set_visible(False)
        ds='HotpotQA' if cell['dataset']=='hotpot' else 'MuSiQue'
        ax.set_title(ds+' / '+cell['proposal'],fontsize=11,fontweight='bold',pad=9)
        c=cell['comparison'];lo,hi=c['ci95_points'];x=.265+col*.47
        fig.text(x,[.600,.155][row],f"{c['wins']} better · {c['ties']} tied · {c['losses']} worse",ha='center',fontsize=9.5)
        fig.text(x,[.570,.125][row],f"ΔF1 {c['delta_points']:+.2f}; 95% CI [{lo:+.2f}, {hi:+.2f}]",ha='center',fontsize=9.5)
    cb=fig.colorbar(im,cax=fig.add_axes([.24,.067,.55,.018]),orientation='horizontal',ticks=[-100,-50,0,50,100])
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=10,length=2);cb.solids.set_rasterized(False)
    fig.text(.515,.012,'Per-question ΔF1 (retention − matched knapsack)',ha='center',fontsize=10)
    save(fig,'fig4_retention_mosaic','Question-level effects at exactly matched lengths')

def policy():
    fig=plt.figure(figsize=(W,3.85))
    for j,kind in enumerate(['fill','anchor']):
        ax=fig.add_axes([.10+j*.51,.18,.36,.57])
        ax.axhline(0,lw=.8,color=GRAY,ls='--');ax.axvline(0,lw=.6,color=LIGHT)
        for r,(color,marker) in zip(D['perturbations'],STYLES):
            x,y=r[kind+'_extra'],r[kind+'_delta'];lo,hi=r[kind+'_ci']
            ax.annotate('',(x,y),(0,0),arrowprops={'arrowstyle':'->','lw':1,'color':color,'alpha':.55})
            ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt=marker,color=color,mfc='white',ms=5.5,capsize=3,elinewidth=1,zorder=4)
        ax.set_title('(a) Continue filling' if j==0 else '(b) Add anchor',fontsize=11,fontweight='bold',pad=9)
        ax.set_xlabel('Extra evidence (Qwen tokens)',fontsize=10,labelpad=5);ax.set_ylabel('ΔF1 (points)',fontsize=10.5)
        ax.grid(axis='y',color=LIGHT,lw=.5)
        ax.set(xlim=(-12,380),ylim=(-9,2),xticks=[0,100,200,300],yticks=[-8,-6,-4,-2,0,2]) if j==0 else ax.set(xlim=(-1.5,17),ylim=(-3.2,3),xticks=[0,5,10,15],yticks=[-3,-2,-1,0,1,2,3])
    handles=[Line2D([],[],color=c,marker=m,mfc='white',lw=0,ms=5.5,label=l) for (c,m),l in zip(STYLES,['Hotpot / Qwen','Hotpot / Phi','MuSiQue / Qwen','MuSiQue / Phi'])]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,frameon=False,fontsize=10,columnspacing=2.5)
    save(fig,'fig5_policy_effects','Evidence quantity and answer quality under policy changes')

def paths():
    fig=plt.figure(figsize=(W,4.8));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def node(x,y,t,active,color):
        ax.add_patch(FancyBboxPatch((x-.09,y-.04),.18,.08,boxstyle='round,pad=0.004,rounding_size=.01',
                     ec=color if active else '#ABB5BB',fc='#F1F8F7' if active else '#F6F7F8',lw=1,linestyle='solid' if active else '--'))
        ax.text(x,y,t,ha='center',va='center',fontsize=10,color=INK if active else GRAY,linespacing=1.15)
    def edge(x0,x1,y,t,active,color):
        ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle='-|>',mutation_scale=10,lw=1.2,color=color if active else '#ADB7BD',linestyle='solid' if active else '--'))
        ax.text((x0+x1)/2,y+.049,t,ha='center',va='bottom',fontsize=9.5,color=INK if active else GRAY,linespacing=1.0)
    for case,top in [(0,.98),(1,.51)]:
        ax.text(.025,top,'(a) Birth year (success)' if case==0 else '(b) Death year (failure)',fontsize=11.5,fontweight='bold',va='top')
        for row,(method,color) in enumerate([('JQR',TEAL),('Reranker',NAVY)]):
            y=[.82,.36][case]-row*.18;active=(method=='JQR') if case==0 else (method=='Reranker')
            ax.text(.073,y,method,ha='center',va='center',fontweight='bold',fontsize=10,color=color)
            key='quote_anchor' if method=='JQR' else 'qwen_rerank_sentence'
            ax.text(.073,y-.049,f"{D['cases'][case]['tokens'][key]} tokens",ha='center',fontsize=9.5,color=GRAY)
            texts=['Ship-tarring\naccount','Pierre Belon','1517\n(birth year)'] if case==0 else ['Bucephalus','Alexander',"323 BC\n(owner's death)"]
            for x,t,act in zip([.25,.505,.76],texts,[True,True,active]):node(x,y,t,act,color)
            edge(.345,.410,y,'described by\n[10]' if case==0 else 'owner [7]',True,color)
            edge(.600,.665,y,'biography [14]' if case==0 else 'death date [13]',active,color)
            answer=('1517' if case==0 else '323 BC') if active else '∅'
            ax.text(.94,y,answer,ha='center',va='center',fontsize=10.5,fontweight='bold',color=color if active else GRAY)
    ax.text(.025,.090,'JQR also retains the horse’s death: “Bucephalus had died ... in 326 BC.”',fontsize=9.5,color=RED,ha='left')
    handles=[Line2D([],[],color=TEAL,lw=1.3,label='Retained'),Line2D([],[],color=GRAY,lw=1,ls='--',label='Omitted'),Line2D([],[],color='white',lw=0,label='∅ Answer unavailable')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=9.5,columnspacing=1.5)
    save(fig,'fig7_evidence_paths','Source-grounded recovery and omission examples')

if __name__=='__main__':
    hops();budgets();raincloud();flows();mosaics();policy();paths()
    REPORT['data_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'figure_data.json',HERE/'budget_sweep_summary.json']}
    (ROOT/'FIGURE_READABILITY_AUDIT.json').write_text(json.dumps(REPORT,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in REPORT.items() if k!='data_sha256'},indent=2))
