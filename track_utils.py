import numpy as np
import trajectory_planning_helpers as tph
import matplotlib.pyplot as plt

def prep_track(reftrack_imp: np.ndarray,
               reg_smooth_opts: dict,
               stepsize_opts: dict) -> tuple:
    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # smoothing and interpolating reference track
    reftrack_interp = tph.spline_approximation. \
        spline_approximation(track=reftrack_imp,
                             k_reg=reg_smooth_opts["k_reg"],
                             s_reg=reg_smooth_opts["s_reg"],
                             stepsize_prep=stepsize_opts["stepsize_prep"],
                             stepsize_reg=stepsize_opts["stepsize_reg"])

    # calculate splines
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
        calc_splines(path=refpath_interp_cl)
    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp

def import_track(file_path: str,
                 width_veh: float) -> np.ndarray:

    # load data from csv file
    csv_data_temp = np.loadtxt(file_path, comments='#', delimiter=',')

    # get coords and track widths out of array
    if np.shape(csv_data_temp)[1] == 3:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2] / 2
        w_tr_l = w_tr_r

    elif np.shape(csv_data_temp)[1] == 4:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2]
        w_tr_l = csv_data_temp[:, 3]

    elif np.shape(csv_data_temp)[1] == 5:  # omit z coordinate in this case
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 3]
        w_tr_l = csv_data_temp[:, 4]

    else:
        raise IOError("Track file cannot be read!")


    # assemble to a single array
    reftrack_imp = np.column_stack((refline_, w_tr_r, w_tr_l))
    # check minimum track width for vehicle width plus a small safety margin
    w_tr_min = np.amin(reftrack_imp[:, 2] + reftrack_imp[:, 3])

    if w_tr_min < width_veh + 0.5:
        print("WARNING: Minimum track width %.2fm is close to or smaller than vehicle width!" % np.amin(w_tr_min))

    return reftrack_imp

def result_plots(width_veh_opt: float,
                 width_veh_real: float,
                 refline: np.ndarray,
                 bound1_interp: np.ndarray,
                 bound2_interp: np.ndarray,
                 trajectory: np.ndarray) -> None:

    # calculate vehicle boundary points (including safety margin in vehicle width)
    normvec_normalized_opt = tph.calc_normal_vectors.\
        calc_normal_vectors(trajectory[:, 3])

    veh_bound1_virt = trajectory[:, 1:3] + normvec_normalized_opt * width_veh_opt / 2
    veh_bound2_virt = trajectory[:, 1:3] - normvec_normalized_opt * width_veh_opt / 2

    veh_bound1_real = trajectory[:, 1:3] + normvec_normalized_opt * width_veh_real / 2
    veh_bound2_real = trajectory[:, 1:3] - normvec_normalized_opt * width_veh_real / 2

    point1_arrow = refline[0]
    point2_arrow = refline[3]
    vec_arrow = point2_arrow - point1_arrow

    # plot track including optimized path
    plt.figure()
    plt.plot(refline[:, 0], refline[:, 1], "k--", linewidth=0.7)
    # plt.plot(veh_bound1_virt[:, 0], veh_bound1_virt[:, 1], "b", linewidth=0.5)
    # plt.plot(veh_bound2_virt[:, 0], veh_bound2_virt[:, 1], "b", linewidth=0.5)
    # plt.plot(veh_bound1_real[:, 0], veh_bound1_real[:, 1], "c", linewidth=0.5)
    # plt.plot(veh_bound2_real[:, 0], veh_bound2_real[:, 1], "c", linewidth=0.5)
    plt.plot(bound1_interp[:, 0], bound1_interp[:, 1], "k-", linewidth=0.7)
    plt.plot(bound2_interp[:, 0], bound2_interp[:, 1], "k-", linewidth=0.7)
    plt.plot(trajectory[:, 1], trajectory[:, 2], "r-", linewidth=1)
    plt.legend(["Centre line",  "Inner track bound", "Outer track bound", "Centreline of Racecar"])

    plt.grid()
    ax = plt.gca()
    ax.arrow(point1_arrow[0], point1_arrow[1], vec_arrow[0], vec_arrow[1],
                head_width=7.0, head_length=7.0, fc='g', ec='g')
    ax.set_aspect("equal", "datalim")
    plt.xlabel("east in m")
    plt.ylabel("north in m")
    plt.show()

    
    # plot curvature profile
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 4])
    plt.grid()
    plt.xlabel("distance in m")
    plt.ylabel("curvature in rad/m")
    plt.show()

# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass