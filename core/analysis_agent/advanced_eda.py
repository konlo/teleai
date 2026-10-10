"""Bounded multivariable EDA with explicit plotted population receipts."""
from dataclasses import asdict
from io import BytesIO
from uuid import uuid4
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter
from core.analysis_agent.analysis_extensions import scoped_dataset
from utils.analysis_datasets import project_dataset,stored_dataset_digest
from utils.analysis_charts import ChartPreview,_apply_unicode_font

KINDS={'correlation_heatmap','scatter_matrix','distribution_panels'}

def render(context,dataset_id,columns,kind,max_points=2000):
    if kind not in KINDS or not 2<=len(columns)<=6 or not 1<=max_points<=5000:
        raise ValueError('Invalid advanced EDA specification')
    parent=scoped_dataset(context,dataset_id,columns)
    frame=project_dataset(context.datasets,parent.id,columns)
    if any(not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]) for c in columns):
        raise ValueError('Advanced EDA requires actual numeric columns; prepare conversions explicitly')
    numeric=frame.astype(float).replace([np.inf,-np.inf],np.nan)
    if len(numeric)==0 or any(numeric[c].notna().sum()<2 for c in columns):raise ValueError('Insufficient finite observations')
    n=len(columns);fig=Figure(figsize=(max(6,n*2.1),max(4,n*1.8)))
    summary={'missing_counts':numeric.isna().sum().to_dict(),'total_rows':len(frame),
             'input_digest':stored_dataset_digest(context.datasets,parent.id)}
    sampled=False
    if kind=='correlation_heatmap':
        corr=numeric.corr();counts=numeric.notna().astype(int).T.dot(numeric.notna().astype(int))
        ax=fig.subplots();im=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(n),columns,rotation=35,ha='right');ax.set_yticks(range(n),columns)
        for i in range(n):
            for j in range(n):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center')
        colorbar=fig.colorbar(im,ax=ax,label='Pearson correlation')
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:format(value,'.2g')))
        summary.update(correlation=corr.to_dict(),pairwise_valid_counts=counts.to_dict())
    elif kind=='scatter_matrix':
        plotted=numeric.sample(n=max_points,random_state=42) if len(numeric)>max_points else numeric
        sampled=len(plotted)<len(numeric);axes=fig.subplots(n,n,squeeze=False)
        for i,y in enumerate(columns):
            for j,x in enumerate(columns):
                ax=axes[i,j]
                if i==j:ax.hist(plotted[x].dropna(),bins=20)
                else:ax.scatter(plotted[x],plotted[y],s=4,alpha=.4)
                if i==n-1:ax.set_xlabel(x)
                if j==0:ax.set_ylabel(y)
        summary.update(plotted_rows=len(plotted),sampling_seed=42 if sampled else None)
    else:
        axes=fig.subplots(n,1,squeeze=False)
        bins={}
        for i,c in enumerate(columns):
            counts,edges,_=axes[i,0].hist(numeric[c].dropna(),bins=20,edgecolor='white')
            axes[i,0].set(xlabel=c,ylabel='Frequency')
            bins[c]={'counts':counts.astype(int).tolist(),'edges':edges.tolist()}
        summary['histograms']=bins
    _apply_unicode_font(fig);fig.tight_layout();buffer=BytesIO();FigureCanvasAgg(fig)
    fig.savefig(buffer,format='png',dpi=110)
    scope=f'보유 {parent.rows:,}행 · {parent.coverage} · '+('산점도 표본 '+str(summary['plotted_rows'])+'행' if sampled else '표본화 없음')
    reason={'correlation_heatmap':'수치 컬럼 간 Pearson 상관계수',
            'scatter_matrix':'수치 컬럼별 산점도와 분포',
            'distribution_panels':'수치 컬럼별 빈도 분포'}[kind]
    card=ChartPreview(str(uuid4()),parent.id,kind,reason,kind,tuple(columns),scope,buffer.getvalue(),
                      {'kind':kind,'columns':columns,'summary':summary,'sampled':sampled})
    context.artifacts[card.id]=card
    return {'status':'ready','cards':[{k:v for k,v in asdict(card).items() if k!='image'}],
            'advanced_eda_receipt':{'dataset_id':parent.id,'kind':kind,'columns':columns,
                                    'scope':scope,'sampled':sampled,'statistics':summary},'scope':scope}
