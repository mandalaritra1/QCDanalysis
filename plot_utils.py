def adjust_plot(ax1,title = None, xlabel = None, ylabel = None, xscale = None, yscale = None, minor_ticks = True, xlabel_coord = 0.95, ylabel_coord = 0.9 , show_legend = False):
    if xlabel != None:
        ax1.set_xlabel(xlabel)
    if ylabel != None:
        ax1.set_ylabel(ylabel)
    if minor_ticks == True:
        ax1.minorticks_on()
    if xscale != None:
        ax1.set_xscale(xscale)
    if yscale != None:
        ax1.set_yscale(yscale)
        
    if title != None:
        ax1.set_title(title)
    ax1.tick_params(axis = 'x', top = True, direction = 'in', length = 5)
    ax1.tick_params(axis = 'y', right = True, direction = 'in', length = 5)
    ax1.tick_params(axis = 'x', which = 'minor', direction = 'in', top = True, length = 2.5)
    ax1.tick_params(axis = 'y', which = 'minor', direction = 'in', right = True, length = 2.5)
    ax1.xaxis.set_label_coords(xlabel_coord, -0.06)
    ax1.yaxis.set_label_coords(-0.1, ylabel_coord)
    if show_legend == True:
        ax1.legend()