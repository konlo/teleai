"""Presentation of verified categorical counts, without changing their data."""
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
import numpy as np


def draw_categories(ax, labels, counts, *, legend=False, stacked=False, palette='default'):
    if palette not in {'default','high_contrast'}:
        raise ValueError('Unknown chart palette')
    base=(['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#000000']
          if palette=='high_contrast' else
          [to_hex(c) for c in colormaps['tab10'].colors])
    colors=(base[:len(labels)] if len(labels)<=len(base) else
            [to_hex(colormaps['turbo'](v)) for v in np.linspace(.05,.95,len(labels))])
    if stacked:
        bottom=0
        for label,count,color in zip(labels,counts,colors):
            ax.bar([0],[count],bottom=bottom,color=color,edgecolor='white',label=label)
            bottom+=count
        ax.set_xticks([0],['Total'])
    else:
        ax.bar(range(len(labels)),counts,color=colors,edgecolor='white')
        ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right')
    if legend:
        ax.legend(handles=[Patch(facecolor=c,label=l) for l,c in zip(labels,colors)])
    return {'legend':legend,'legend_labels':labels if legend else [],
            'stacked':stacked,'palette':palette,'colors':colors}
