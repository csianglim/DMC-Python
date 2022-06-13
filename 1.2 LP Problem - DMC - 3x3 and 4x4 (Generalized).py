# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Linear Programming: DMC Formulation (3x3)
# - Author: Siang Lim, Shams Elnawawi
# - Last Updated: June 7th 2022
# - Created March 2nd 2022
#
# ## References
# - Morshedi, A. M., Cutler, C. R., & Skrovanek, T. A. (1985). Optimal solution of dynamic matrix control with linear programing techniques (LDMC). In 1985 American Control Conference (pp. 199-208). IEEE.
# - Sorensen, R. C., & Cutler, C. R. (1998). LP integrates economics into dynamic matrix control. Hydrocarbon Processing, 77(9), 57-65.
# - Ranade, S. M., & Torres, E. (2009). From dynamic mysterious control to dynamic manageable control. Hydrocarbon Processing, 88(3), 77-81.
# - Godoy, J. L., Ferramosca, A., & González, A. H. (2017). Economic performance assessment and monitoring in LP-DMC type controller applications. Journal of Process Control, 57, 26-37.

# +
# # !pip install pulp

# +
# Just importing libraries and tweaking the plot settings
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget

from matplotlib.ticker import StrMethodFormatter
import matplotlib.gridspec as gridspec
from labellines import labelLine, labelLines

from ipywidgets.widgets.interaction import interact
import ipywidgets.widgets as widgets
from ipywidgets import Layout

# Import PuLP modeler functions
from pulp import *

fsize = 8
tsize = 12
tdir = 'in'
major = 5.0
minor = 3.0
lwidth = 0.8
lhandle = 2.0
plt.style.use('default')
plt.rcParams['font.size'] = fsize
plt.rcParams['legend.fontsize'] = tsize
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.linewidth'] = lwidth
plt.rcParams['legend.handlelength'] = lhandle
# -

# ## MV-CV equations
#
# $$
# G =
#     \begin{bmatrix}
#         -0.200 & -0.072 & 0.0774 \\
#         0.125 & -0.954 &  0.0063 \\
#         0.025 & 0.101 & −0.0143
#     \end{bmatrix}
# $$
#
# Using the gain matrix, the CV relationship can be written in terms of its MVs, starting with:
#
# $$
# \Delta \text{CV}_{1} = G_{11} \Delta \text{MV}_{1} + G_{12} \Delta \text{MV}_{2} + G_{13} \Delta \text{MV}_{3} \\ 
# \Delta \text{CV}_{2} = G_{21} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2} + G_{23} \Delta \text{MV}_{3} \\
# \Delta \text{CV}_{3} = G_{31} \Delta \text{MV}_{1} + G_{32} \Delta \text{MV}_{2} + G_{33} \Delta \text{MV}_{3}
# $$
#
# We can impose upper and lower limits on the MVs:
#
# $$
# \text{MV}_{1, \text{Lo}} \leq \text{MV}_{1} \leq \text{MV}_{1, \text{Hi}}\\ 
# \text{MV}_{2, \text{Lo}} \leq \text{MV}_{2} \leq \text{MV}_{2, \text{Hi}}\\
# \text{MV}_{3, \text{Lo}} \leq \text{MV}_{3} \leq \text{MV}_{3, \text{Hi}}\\
# $$
#
# As well as the CVs:
#
# $$
# \text{CV}_{1, \text{Lo}} \leq \text{CV}_{1} \leq \text{CV}_{1, \text{Hi}}\\ 
# \text{CV}_{2, \text{Lo}} \leq \text{CV}_{2} \leq \text{CV}_{2, \text{Hi}}\\
# \text{CV}_{3, \text{Lo}} \leq \text{CV}_{3} \leq \text{CV}_{3, \text{Hi}}\\
# $$
#
# Since the CVs are related to the MVs by the gain matrix, we can substitute the equations to get CV limits in terms of MV movements:
#
# $$
# G_{11} \Delta \text{MV}_{1} + G_{12} \Delta \text{MV}_{2} + G_{13} \Delta \text{MV}_{3} \leq \Delta \text{CV}_{1, \text{Hi}}\\
# G_{11} \Delta \text{MV}_{1} + G_{12} \Delta \text{MV}_{2} + G_{13} \Delta \text{MV}_{3} \geq \Delta \text{CV}_{1, \text{Lo}}\\
# G_{21} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2} + G_{23} \Delta \text{MV}_{3} \leq \Delta \text{CV}_{2, \text{Hi}}\\
# G_{21} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2} + G_{23} \Delta \text{MV}_{3} \geq \Delta \text{CV}_{2, \text{Lo}}\\
# G_{31} \Delta \text{MV}_{1} + G_{32} \Delta \text{MV}_{2} + G_{33} \Delta \text{MV}_{3} \leq \Delta \text{CV}_{3, \text{Hi}}\\
# G_{31} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2} + G_{33} \Delta \text{MV}_{3} \geq \Delta \text{CV}_{3, \text{Lo}}\\
# $$
#
# Let's see what this looks like:

# # Gains and CV Limits

# +
G11 = -0.200
G12 = -0.072
G13 = 0.0774
G14 = 0.0574

G21 = 0.125
G22 = -0.954
G23 = 0.0063
G24 = 0.0374

G31 = 0.025
G32 = 0.101
G33 = -0.0143
G34 = -0.1143

G41 = 0.120
G42 = 0.150
G43 = -0.1143
G44 = -0.0943

CV1Lo = -6
CV1Hi = 6

CV2Lo = -10
CV2Hi = 10.5

CV3Lo = -3
CV3Hi = 3.5

CV4Lo = -4.5
CV4Hi = 4

cost_MV1 = -1
cost_MV2 = -1
cost_MV3 = -1
cost_MV4 = 1

limits = 10
plot_limits = 20

MV1Lo = -limits
MV1Hi = limits
MV2Lo = -limits
MV2Hi = limits
MV3Lo = -limits
MV3Hi = 15
MV4Lo = -limits
MV4Hi = -limits

# +
G           = [(G11, G12, G13, G14), 
               (G21, G22, G23, G24), 
               (G31, G32, G33, G34), 
               (G41, G42, G43, G44)]

CV_values   = [(CV1Lo, CV1Hi), 
               (CV2Lo, CV2Hi), 
               (CV3Lo, CV3Hi)] 
#                (CV4Lo, CV4Hi)]
CV_init_vals= [(-2.5,2.0), 
               (-4,4.5), 
               (-0.9,0.2)] 
#                (-4,4.5)]
MV_costs    = [cost_MV1, 
               cost_MV2, 
               cost_MV3]
#                cost_MV4]
MV_values   = [(MV1Lo, MV1Hi), 
               (MV2Lo, MV2Hi), 
               (MV3Lo, MV3Hi)] 
#                (MV4Lo, MV4Hi)]

# (x,y,c) triplets of MVs, c for constant, i.e. 
# (MV1, MV2, MV3) for the first plot, 
# (MV1, MV3, MV2) for the second, 
# (MV2, MV3, MV1) for the third
# plot_MV_indices = [(0,1), (0,2), (1,2)]

nCVs = len(CV_values)
nMVs = len(MV_values)

# +
CV_widgets   = []
MV_widgets   = []
cost_widgets = []
stepsize = 0.2

# Make CV sliders
for i in range(nCVs):
    widget = widgets.FloatRangeSlider(
                value=CV_init_vals[i], min=CV_values[i][0], max=CV_values[i][1], step=stepsize,
                description=f'CV{i+1} Limits', continuous_update=False)
    CV_widgets.append(widget)

# Make MV sliders
for i in range(nMVs):
    widget = widgets.FloatRangeSlider(
                value=MV_values[i], min=MV_values[i][0], max=MV_values[i][1], step=stepsize,
                description=f'MV{i+1} Limits', continuous_update=False)
    MV_widgets.append(widget)
    
# Make cost sliders
for i in range(nMVs):
    widget = widgets.FloatSlider(
                value=MV_costs[i], min=-3, max=3, step=stepsize/2, # finer step for costs
                description=f'MV{i+1} Cost', continuous_update=True)
    cost_widgets.append(widget)
    
sliders = CV_widgets + cost_widgets + MV_widgets
values = [slider.value for slider in sliders]
values_dict = {}
for slider in sliders:
    values_dict[slider.description] = slider.value
print(values_dict)

# +
# display ui
ncols = 3

nrows = nCVs // ncols + 1
CV_row = []
for i in range(nrows):
    CV_row.append(widgets.VBox(CV_widgets[(ncols)*i:(ncols)*i+ncols]))
    
nrows = nMVs // ncols + 1
MV_row = []
cost_row = []
for i in range(nrows):
    MV_row.append(widgets.VBox(MV_widgets[(ncols)*i:(ncols)*i+ncols]))
    cost_row.append(widgets.VBox(cost_widgets[(ncols)*i:(ncols)*i+ncols]))

ui = widgets.VBox([widgets.VBox(CV_row),
              widgets.VBox(MV_row),
              widgets.VBox(cost_row)])


# +
# Solve the LP
def run_lp(values_dict):
    prob = LpProblem("DMC_problem",LpMinimize)
    
    # How many MVs
    MVs = []
    for i in range(nMVs):
        MVs.append(LpVariable(f"MV{i+1}",-limits))
    
    # the objective function
    obj = 0
    for indx, MV in enumerate(MVs):
        obj += values_dict[f"MV{indx+1} Cost"]*MV
        
    prob += obj, "Cost function of MVs"
    
    # constraint formulation in terms of MV1 and MV2
    CV_contraint_lo = []
    CV_contraint_hi = []
    
    for i in range(nCVs):
        c = 0
        for indx, MV in enumerate(MVs):
            c += G[i][indx]*MV
        prob += c <= values_dict[f'CV{i+1} Limits'][1], f'CV{i+1} High Limit'
        prob += c >= values_dict[f'CV{i+1} Limits'][0], f'CV{i+1} Low Limit'
    
#     prob += G[i][0]*MV1 + G[i][1]*MV2 <= values_dict[f'CV{i+1} Limits'][1], f'CV{i+1} High Limit'
#     prob += G[i][0]*MV1 + G[i][1]*MV2 >= values_dict[f'CV{i+1} Limits'][0], f'CV{i+1} Low Limit'
    
    for indx, MV in enumerate(MVs):
        prob += MV <= values_dict[f'MV{indx+1} Limits'][1], f'MV{indx+1} High Limit'
        prob += MV >= values_dict[f'MV{indx+1} Limits'][0], f'MV{indx+1} Low Limit'        
    
    if (prob.solve(PULP_CBC_CMD(msg=0)) == 1):
#         print([v.varValue for v in prob.variables()])
#         print([v for v in prob.variables()])
        return [v.varValue for v in prob.variables()], value(prob.objective)
    else:
        print("NOT SOLVED - Infeasibility!")
        return np.zeros(nMVs), 0


# +
d = np.linspace(-plot_limits, plot_limits, 1000)
x,y = np.meshgrid(d,d)

# Recalculate shaded regions
constraints = []
for i in range(nCVs):
    c_hi = G[i][0]*x + G[i][1]*y <= values_dict[f'CV{i+1} Limits'][1]
    c_lo = G[i][0]*x + G[i][1]*y >= values_dict[f'CV{i+1} Limits'][0]
    constraints.append(c_lo)
    constraints.append(c_hi)

# the MVs
for indx, var in enumerate([x,y]):
    c_hi = var <= values_dict[f'MV{indx+1} Limits'][1]
    c_lo = var >= values_dict[f'MV{indx+1} Limits'][0]
    constraints.append(c_lo)
    constraints.append(c_hi)


# -

def handle_slider_change(change):
    ## grab slider vals
    values_dict = {}
    for slider in sliders:
        values_dict[slider.description] = slider.value
    
    # Find the LP soln
    soln, V = run_lp(values_dict)
    
    for indx, key in enumerate(axs_dict):
        ax = axs_dict[key]
        
        # remove previous line labels (TODO could be more efficient, modify labelLine library to return handles)
        # for txt in ax.texts:
        #     txt.remove()
        
        # figure out which MVs we are plotting, and which ones are constant
        x_MV = key[1] # column is x-axis
        y_MV = key[0]+1 # row is y-axis
        c_MV = [a for a in range(nMVs) if a != x_MV and a != y_MV]        

        # Recalculate shaded regions 
        constraints = []
        
        # the obj func
        y_obj = (V - sum([values_dict[f'MV{cmv+1} Cost']*soln[cmv] for cmv in c_MV]) - values_dict[f'MV{x_MV+1} Cost']*d)/values_dict[f'MV{y_MV+1} Cost']
                
        # Update CV constraint lines
        for i in range(nCVs):
            cv_lim_lo = values_dict[f'CV{i+1} Limits'][0]
            cv_lim_hi = values_dict[f'CV{i+1} Limits'][1]

            # for CV constraint lines
            y_lo = (cv_lim_lo - G[i][x_MV]*d - sum([G[i][cmv]*soln[cmv] for cmv in c_MV]))/G[i][y_MV]
            y_hi = (cv_lim_hi - G[i][x_MV]*d - sum([G[i][cmv]*soln[cmv] for cmv in c_MV]))/G[i][y_MV]
            lines_handler_dict[key]['CV_lines_lo'][i].set_data(d, y_lo)
            lines_handler_dict[key]['CV_lines_hi'][i].set_data(d, y_hi)

            # for shading
            c_hi = G[i][x_MV]*x + G[i][y_MV]*y + sum([G[i][cmv]*soln[cmv] for cmv in c_MV]) <= cv_lim_hi
            c_lo = G[i][x_MV]*x + G[i][y_MV]*y + sum([G[i][cmv]*soln[cmv] for cmv in c_MV]) >= cv_lim_lo
            constraints.append(c_lo)
            constraints.append(c_hi)
            
            # search for valid xvals where the yvals are within plots limits
            # x_valid_lo = d[(y_lo < plot_limits) & (y_lo > -plot_limits)]
            # x_valid_hi = d[(y_hi < plot_limits) & (y_hi > -plot_limits)]
            # offset_lo = max(0,min(10, len(x_valid_lo)-3))
            # offset_hi = max(0,min(10, len(x_valid_hi)-3))
            # if len(x_valid_lo) > 0:
            #     txt_cv_lo = labelLine(line_lo, x_valid_lo[offset_lo], fontsize=5, zorder=2.5)
            # if len(x_valid_hi) > 0:
            #     txt_cv_hi = labelLine(line_hi, x_valid_hi[-offset_hi], fontsize=5, zorder=2.5)            

        # Update MV constraint lines
        lines_handler_dict[key]['v_lo'].set_data([values_dict[f'MV{x_MV+1} Limits'][0],values_dict[f'MV{x_MV+1} Limits'][0]], [-limits, limits])
        lines_handler_dict[key]['v_hi'].set_data([values_dict[f'MV{x_MV+1} Limits'][1],values_dict[f'MV{x_MV+1} Limits'][1]], [-limits, limits])
        lines_handler_dict[key]['h_lo'].set_data([-limits, limits], [values_dict[f'MV{y_MV+1} Limits'][0],values_dict[f'MV{y_MV+1} Limits'][0]])
        lines_handler_dict[key]['h_hi'].set_data([-limits, limits], [values_dict[f'MV{y_MV+1} Limits'][1],values_dict[f'MV{y_MV+1} Limits'][1]])    

        # the 4 MV limits for this ax
        constraints.append(x >= values_dict[f'MV{x_MV+1} Limits'][0])
        constraints.append(x <= values_dict[f'MV{x_MV+1} Limits'][1])
        constraints.append(y >= values_dict[f'MV{y_MV+1} Limits'][0])
        constraints.append(y <= values_dict[f'MV{y_MV+1} Limits'][1]) 

        # Shade the right regions
        lines_handler_dict[key]['im'].set_data((np.logical_and.reduce(constraints)).astype(float))
        
        # the quiver/vector field
        z_obj = (values_dict[f'MV{x_MV+1} Cost'] * xv) + (values_dict[f'MV{y_MV+1} Cost'] * yv) + sum([values_dict[f'MV{cmv+1} Cost']*soln[cmv] for cmv in c_MV])
        lines_handler_dict[key]['quiver'].set_UVC(-values_dict[f'MV{x_MV+1} Cost'],-values_dict[f'MV{y_MV+1} Cost'], z_obj)

        # the soln
        lines_handler_dict[key]['soln_marker'].set_data(soln[x_MV], soln[y_MV]);
        lines_handler_dict[key]['soln_text'].set_position((soln[x_MV], soln[y_MV]));
        lines_handler_dict[key]['soln_text'].set_text("({:.1f}, {:.1f})".format(soln[x_MV], soln[y_MV]))
        lines_handler_dict[key]['soln_func'].set_data(d, y_obj);
    
    fig.canvas.draw()


# +
# Initialize plots
d = np.linspace(-plot_limits, plot_limits, 100)
x,y = np.meshgrid(d,d)

gs = gridspec.GridSpec(nMVs-1, nMVs-1)
gs.update(wspace=0.05, hspace=0.05)
gs_indices = [(row, col) for row in range(nMVs-2,-1,-1) for col in range(row+1)]
axs_dict = {}

# mesh for vector field
dvec = np.linspace(-plot_limits + (0.1*plot_limits), plot_limits - (0.1*plot_limits), 12)
xv, yv = np.meshgrid(dvec, dvec)

# init soln
soln, V = run_lp(values_dict)

# initialize line handlers
CV_lines_lo = []
CV_lines_hi = []
MV_lines_lo = []
MV_lines_hi = []
lines_handler_dict = {}

colors = ['r', 'b', 'y', 'g'] # TODO generalize this using a cmap instead of defining manual colors??!

# plot as widget
output = widgets.Output()
with output:
    fig = plt.figure(figsize=(6,6), dpi=100, facecolor='white')
    plt.show()
    
for r,c in gs_indices:
    # build the shared axes correctly and handle the tick labels
    if (r == nMVs-2 and c == 0): # no shared axis for the ax, bottom left corner
        ax = plt.subplot(gs[r,c])
    elif(c != 0): # all non-first columns share a y-axis with stuff to the left of it
        ax = plt.subplot(gs[r,c], sharey=axs_dict[(r,0)])
        plt.setp(ax.get_yticklabels(), visible=False)
    elif(r != nMVs-2): # all non-first rows share a x-axis with stuff below it
        ax = plt.subplot(gs[r,c], sharex=axs_dict[(nMVs-2,c)])
        plt.setp(ax.get_xticklabels(), visible=False)

    # add the labels
    if(r == nMVs-2):
        ax.set_xlabel(f'$\Delta MV_{c+1}$')
    if(c == 0):
        ax.set_ylabel(f'$\Delta MV_{r+2}$')

    axs_dict[(r,c)] = ax

for indx, key in enumerate(axs_dict):
    ax = axs_dict[key]
    lines_handler_dict[key] = {}

    # figure out which MVs we are plotting, and which ones are constant
    x_MV = key[1] # column is x-axis
    y_MV = key[0]+1 # row is y-axis
    c_MV = [a for a in range(nMVs) if a != x_MV and a != y_MV]

    # plot the MV limits
    v_lo = ax.axvline(x=values_dict[f'MV{x_MV+1} Limits'][0], color='gray', lw=1, label=f'MV{x_MV+1} Lo')
    v_hi = ax.axvline(x=values_dict[f'MV{x_MV+1} Limits'][1], color='gray', lw=1, label=f'MV{x_MV+1} Hi')
    h_lo = ax.axhline(y=values_dict[f'MV{y_MV+1} Limits'][0], color='gray', lw=1, label=f'MV{y_MV+1} Lo')
    h_hi = ax.axhline(y=values_dict[f'MV{y_MV+1} Limits'][1], color='gray', lw=1, label=f'MV{y_MV+1} Hi')

    # labelLines([v_lo, v_hi], fontsize=4)
    # labelLine(h_hi, fontsize=4)
        
    # store the line handlers per ax in a dict
    lines_handler_dict[key]['v_lo'] = v_lo
    lines_handler_dict[key]['v_hi'] = v_hi
    lines_handler_dict[key]['h_lo'] = h_lo
    lines_handler_dict[key]['h_hi'] = h_hi

    # Recalculate shaded regions 
    constraints = []

    # calculate CV lines
    lines_handler_dict[key]['CV_lines_lo'] = []
    lines_handler_dict[key]['CV_lines_hi'] = []    
    for i in range(nCVs):
        cv_lim_lo = values_dict[f'CV{i+1} Limits'][0]
        cv_lim_hi = values_dict[f'CV{i+1} Limits'][1]

        # for CV constraint lines
        y_lo = (cv_lim_lo - G[i][x_MV]*d - sum([G[i][cmv]*soln[cmv] for cmv in c_MV]))/G[i][y_MV]
        y_hi = (cv_lim_hi - G[i][x_MV]*d - sum([G[i][cmv]*soln[cmv] for cmv in c_MV]))/G[i][y_MV]
        line_lo, = ax.plot(d, y_lo, f'--{colors[i]}', label=f'$CV_{i+1}$ Lo');
        line_hi, = ax.plot(d, y_hi, f'-{colors[i]}', label=f'$CV_{i+1}$ Hi');
        lines_handler_dict[key]['CV_lines_lo'].append(line_lo)
        lines_handler_dict[key]['CV_lines_hi'].append(line_hi)
        
        # # search for valid xvals where the yvals are within plots limits
        # x_valid_lo = d[(y_lo < plot_limits) & (y_lo > -plot_limits)]
        # x_valid_hi = d[(y_hi < plot_limits) & (y_hi > -plot_limits)]
        # offset_lo = max(0,min(10, len(x_valid_lo)-3))
        # offset_hi = max(0,min(10, len(x_valid_hi)-3))
        # if len(x_valid_lo) > 0:
        #     labelLine(line_lo, x_valid_lo[offset_lo], fontsize=5, zorder=2.5)
        # if len(x_valid_hi) > 0:
        #     labelLine(line_hi, x_valid_hi[-offset_hi], fontsize=5, zorder=2.5)
        
        # for shading
        c_hi = G[i][x_MV]*x + G[i][y_MV]*y + sum([G[i][cmv]*soln[cmv] for cmv in c_MV]) <= cv_lim_hi
        c_lo = G[i][x_MV]*x + G[i][y_MV]*y + sum([G[i][cmv]*soln[cmv] for cmv in c_MV]) >= cv_lim_lo
        constraints.append(c_lo)
        constraints.append(c_hi)
            

    # the 4 MV limits for this ax
    constraints.append(x >= values_dict[f'MV{x_MV+1} Limits'][0])
    constraints.append(x <= values_dict[f'MV{x_MV+1} Limits'][1])
    constraints.append(y >= values_dict[f'MV{y_MV+1} Limits'][0])
    constraints.append(y <= values_dict[f'MV{y_MV+1} Limits'][1])    

    # the obj function        
    y_obj = (V - sum([values_dict[f'MV{cmv+1} Cost']*soln[cmv] for cmv in c_MV]) - values_dict[f'MV{x_MV+1} Cost']*d)/values_dict[f'MV{y_MV+1} Cost']
    soln_func, = ax.plot(d, y_obj, '--k')
    soln_marker, = ax.plot(soln[x_MV], soln[y_MV], 'ok', zorder=10)
    soln_text = ax.text(soln[x_MV], soln[y_MV], '({:.2f},{:.2f})'.format(soln[x_MV], soln[y_MV]))

    lines_handler_dict[key]['soln_func'] = soln_func
    lines_handler_dict[key]['soln_marker'] = soln_marker
    lines_handler_dict[key]['soln_text'] = soln_text
    
    z_obj = (values_dict[f'MV{x_MV+1} Cost'] * xv) + (values_dict[f'MV{y_MV+1} Cost'] * yv) + sum([values_dict[f'MV{cmv+1} Cost']*soln[cmv] for cmv in c_MV])
    lines_handler_dict[key]['quiver'] = ax.quiver(xv, yv,-values_dict[f'MV{x_MV+1} Cost'],-values_dict[f'MV{y_MV+1} Cost'], z_obj, cmap='gray', headwidth=4, width=0.003, scale=40, alpha=0.5)

    ax.plot(0,0,'kx');
    ax.set_aspect('equal')
    im = ax.imshow(np.logical_and.reduce(constraints).astype(int), extent=(x.min(),x.max(),y.min(),y.max()), origin="lower", cmap="Blues", alpha=0.1)
    lines_handler_dict[key]['im'] = im
        
####################################################
# END OF INIT FUNC
####################################################

# register slides
for widget in sliders:
    widget.observe(handle_slider_change, names='value')

widgets.HBox([output, 
              widgets.VBox([widgets.HTMLMath(
                            value=r"<p>Use the controls to interact with this linear program</p><br><br>"), 
                            ui])])

# -


