import numpy as np
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def display_diff(diff, ax=None, lim=1, mid=0, title=None, fontsize=11, norm=None, cmap="bwr"):
	show = False
	if ax is None:
		fig, ax = plt.subplots()
		show = True

	norm_kw = {}
	if norm == "Centered":
		norm_kw.update({"norm": colors.CenteredNorm(vcenter=mid)})
	elif norm == "Log":
		norm_kw.update({"norm": colors.LogNorm()})

	div = make_axes_locatable(ax)
	cax = div.append_axes("right", size="5%", pad=0.1)

	im = ax.imshow(diff, origin='lower', extent=(-lim, lim, -lim, lim),
				   cmap=cmap, **norm_kw)

	plt.colorbar(im, cax=cax)

	tx = None
	if title is not None:
		tx = ax.set_title(title, fontsize=fontsize)

	if show: plt.show()

	return im, tx