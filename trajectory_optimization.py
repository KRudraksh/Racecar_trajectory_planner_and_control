import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
from optimization_utils import opt_min_curv, opt_shortest_path
import configparser
from track_utils import prep_track, import_track, result_plots
from planner_utils import create_raceline, calc_head_curv_an, calc_ax_profile, calc_t_profile
from calc_vel_profile import calc_vel_profile


######################## VARIABLES ###############################

# choose vehicle parameter file ----------------------------------------------------------------------------------------
file_paths = {"veh_params_file": "params.ini"}

# select track file (including centerline coordinates + track widths) --------------------------------------------------
# file_paths["track_name"] = "rounded_rectangle"                       # artificial track
# file_paths["track_name"] = "handling_track"                          # artificial track
file_paths["track_name"] = "berlin_2018"                             # Berlin Formula E 2018
# file_paths["track_name"] = "modena_2019"                             # Modena 2019


# set optimization type ------------------------------------------------------------------------------------------------
# 'shortest_path'       shortest path optimization
# 'mincurv'             minimum curvature optimization without iterative call
opt_type = 'mincurv'

# get current path
file_paths["module"] = os.path.dirname(os.path.abspath(__file__))


###################### PATH INITIALIZATION #########################

# assemble track import path
file_paths["track_file"] = os.path.join(file_paths["module"], "inputs", "tracks", file_paths["track_name"] + ".csv")

# assemble friction map import paths
file_paths["tpamap"] = os.path.join(file_paths["module"], "inputs", "frictionmaps",
                                    file_paths["track_name"] + "_tpamap.csv")

# create outputs folder(s)
os.makedirs(file_paths["module"] + "/outputs", exist_ok=True)

# assemble export paths
file_paths["traj_race_export"] = os.path.join(file_paths["module"], "outputs", "traj_race_cl.csv")


##################### VEHICLE PARAMS ###########################

# load vehicle parameter file into a "pars" dict
parser = configparser.ConfigParser()
pars = {}

if not parser.read(os.path.join(file_paths["module"], "params", file_paths["veh_params_file"])):
    raise ValueError('Specified config file does not exist or is empty!')

pars["ggv_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ggv_file'))
pars["ax_max_machines_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ax_max_machines_file'))
pars["stepsize_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))
pars["reg_smooth_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts'))
pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))

if opt_type == 'shortest_path':
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_shortest_path'))

elif opt_type in ['mincurv', 'mincurv_iqp']:
    pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mincurv'))

file_paths["ggv_file"] = os.path.join(file_paths["module"], "inputs", "veh_dyn_info", pars["ggv_file"])
file_paths["ax_max_machines_file"] = os.path.join(file_paths["module"], "inputs", "veh_dyn_info",
                                                      pars["ax_max_machines_file"])


################################# IMPORT TRACK AND VEHICLE DYNAMICS ###################

# save start time
t_start = time.perf_counter()

# import track
reftrack_imp = import_track(file_path=file_paths["track_file"], width_veh=pars["veh_params"]["width"])

#import ggv file
with open(file_paths["ggv_file"], "rb") as fh:
            ggv = np.loadtxt(fh, comments='#', delimiter=",")
# expand dimension in case of a single row
if ggv.ndim == 1:
    ggv = np.expand_dims(ggv, 0)

#import AX MAX machines
with open(file_paths["ax_max_machines_file"], "rb") as fh:
    ax_max_machines = np.loadtxt(fh, comments='#',  delimiter=",")

# expand dimension in case of a single row
if ax_max_machines.ndim == 1:
    ax_max_machines = np.expand_dims(ax_max_machines, 0)
        
reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = prep_track(reftrack_imp=reftrack_imp,
                                                reg_smooth_opts=pars["reg_smooth_opts"],
                                                stepsize_opts=pars["stepsize_opts"])


############################# OPTIMIZATION ################################
 
# call optimization
if opt_type == 'mincurv':
    alpha_opt = opt_min_curv(reftrack=reftrack_interp,
                                              normvectors=normvec_normalized_interp,
                                              A=a_interp,
                                              kappa_bound=pars["veh_params"]["curvlim"],
                                              w_veh=pars["optim_opts"]["width_opt"])[0]

elif opt_type == 'shortest_path':
    alpha_opt = opt_shortest_path(reftrack=reftrack_interp,
                                                        normvectors=normvec_normalized_interp,
                                                        w_veh=pars["optim_opts"]["width_opt"])

else:
    raise ValueError('Unknown optimization type!')


############################ SPLINE INTERPOLATION ##############################

raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
    spline_lengths_opt, el_lengths_opt_interp = create_raceline(refline=reftrack_interp[:, :2],
                    normvectors=normvec_normalized_interp,
                    alpha=alpha_opt,
                    stepsize_interp=pars["stepsize_opts"]["stepsize_interp_after_opt"])


############################## HEADING AND CURVATURE ##############################

# calculate heading and curvature (analytically)
psi_vel_opt, kappa_opt = calc_head_curv_an(coeffs_x=coeffs_x_opt,
                      coeffs_y=coeffs_y_opt,
                      ind_spls=spline_inds_opt_interp,
                      t_spls=t_vals_opt_interp)


####################### VELOCITY AND ACCELERATION PROFILE #######################

vx_profile_opt = calc_vel_profile(ggv=ggv,
                        ax_max_machines=ax_max_machines,
                        v_max=pars["veh_params"]["v_max"],
                        kappa=kappa_opt,
                        el_lengths=el_lengths_opt_interp,
                        closed=True,
                        filt_window=pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
                        dyn_model_exp=pars["vel_calc_opts"]["dyn_model_exp"],
                        drag_coeff=pars["veh_params"]["dragcoeff"],
                        m_veh=pars["veh_params"]["mass"])

# calculate longitudinal acceleration profile
vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
ax_profile_opt = calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                                     el_lengths=el_lengths_opt_interp,
                                                     eq_length_output=False)

# calculate laptime
t_profile_cl = calc_t_profile(vx_profile=vx_profile_opt,
                                                 ax_profile=ax_profile_opt,
                                                 el_lengths=el_lengths_opt_interp)
print("INFO: Estimated laptime: %.2fs" % t_profile_cl[-1])

s_points = np.cumsum(el_lengths_opt_interp[:-1])
s_points = np.insert(s_points, 0, 0.0)

plt.plot(s_points, vx_profile_opt)
plt.plot(s_points, ax_profile_opt)
plt.plot(s_points, t_profile_cl[:-1])

plt.grid()
plt.xlabel("distance in m")
plt.legend(["vx in m/s", "ax in m/s2", "t in s"])

plt.show()


###################### PLOTTING ###########################
# arrange data into one trajectory
trajectory_opt = np.column_stack((s_points_opt_interp,
                                  raceline_interp,
                                  psi_vel_opt,
                                  kappa_opt,
                                  vx_profile_opt,
                                  ax_profile_opt))
spline_data_opt = np.column_stack((spline_lengths_opt, coeffs_x_opt, coeffs_y_opt))

# print end time
print("INFO: Runtime from import to final trajectory was %.2fs" % (time.perf_counter() - t_start))

# calculate boundaries and interpolate them to small stepsizes (currently linear interpolation)
bound1 = reftrack_interp[:, :2] + normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 2], 1)
bound2 = reftrack_interp[:, :2] - normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 3], 1)

# plot results
result_plots(width_veh_opt=pars["optim_opts"]["width_opt"],
                width_veh_real=pars["veh_params"]["width"],
                refline=reftrack_interp[:, :2],
                bound1_interp=bound1,
                bound2_interp=bound2,
                trajectory=trajectory_opt)
