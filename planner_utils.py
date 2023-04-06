import numpy as np
import math
from typing import Union

def calc_t_profile(vx_profile: np.ndarray,
                   el_lengths: np.ndarray,
                   t_start: float = 0.0,
                   ax_profile: np.ndarray = None) -> np.ndarray:

    # calculate acceleration profile if required
    if ax_profile is None:
        ax_profile = calc_ax_profile(vx_profile=vx_profile,
                                        el_lengths=el_lengths,
                                        eq_length_output=False)

    # calculate temporal duration of every step between two points
    no_points = el_lengths.size
    t_steps = np.zeros(no_points)

    for i in range(no_points):
        if not math.isclose(ax_profile[i], 0.0):
            t_steps[i] = (-vx_profile[i] + math.sqrt((math.pow(vx_profile[i], 2) + 2 * ax_profile[i] * el_lengths[i])))\
                         / ax_profile[i]

        else:  # ax == 0.0
            t_steps[i] = el_lengths[i] / vx_profile[i]

    # calculate temporal duration profile out of steps
    t_profile = np.insert(np.cumsum(t_steps), 0, 0.0) + t_start

    return t_profile

def calc_spline_lengths(coeffs_x: np.ndarray,
                        coeffs_y: np.ndarray,
                        no_interp_points: int = 15) -> np.ndarray:

    # catch case with only one spline
    if coeffs_x.size == 4 and coeffs_x.shape[0] == 4:
        coeffs_x = np.expand_dims(coeffs_x, 0)
        coeffs_y = np.expand_dims(coeffs_y, 0)

    # get number of splines and create output array
    no_splines = coeffs_x.shape[0]
    spline_lengths = np.zeros(no_splines)

    # loop through all the splines and calculate intermediate coordinates
    t_steps = np.linspace(0.0, 1.0, no_interp_points)
    spl_coords = np.zeros((no_interp_points, 2))

    for i in range(no_splines):
        spl_coords[:, 0] = coeffs_x[i, 0] \
                            + coeffs_x[i, 1] * t_steps \
                            + coeffs_x[i, 2] * np.power(t_steps, 2) \
                            + coeffs_x[i, 3] * np.power(t_steps, 3)
        spl_coords[:, 1] = coeffs_y[i, 0] \
                            + coeffs_y[i, 1] * t_steps \
                            + coeffs_y[i, 2] * np.power(t_steps, 2) \
                            + coeffs_y[i, 3] * np.power(t_steps, 3)

        spline_lengths[i] = np.sum(np.sqrt(np.sum(np.power(np.diff(spl_coords, axis=0), 2), axis=1)))

    return spline_lengths

def create_raceline(refline: np.ndarray,
                    normvectors: np.ndarray,
                    alpha: np.ndarray,
                    stepsize_interp: float) -> tuple:
   
    # calculate raceline on the basis of the optimized alpha values
    raceline = refline + np.expand_dims(alpha, 1) * normvectors

    # calculate new splines on the basis of the raceline
    raceline_cl = np.vstack((raceline, raceline[0]))

    coeffs_x_raceline, coeffs_y_raceline, A_raceline, normvectors_raceline = calc_splines(path=raceline_cl,
                     use_dist_scaling=False)

    # calculate new spline lengths
    spline_lengths_raceline = calc_spline_lengths(coeffs_x=coeffs_x_raceline,
                            coeffs_y=coeffs_y_raceline)

    # interpolate splines for evenly spaced raceline points
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = interp_splines(spline_lengths=spline_lengths_raceline,
                                      coeffs_x=coeffs_x_raceline,
                                      coeffs_y=coeffs_y_raceline,
                                      incl_last_point=False,
                                      stepsize_approx=stepsize_interp)

    # calculate element lengths
    s_tot_raceline = float(np.sum(spline_lengths_raceline))
    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])

    return raceline_interp, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, \
           t_values_raceline_interp, s_raceline_interp, spline_lengths_raceline, el_lengths_raceline_interp_cl

def normalize_psi(psi: Union[np.ndarray, float]) -> np.ndarray:
  
    # use modulo operator to remove multiples of 2*pi
    psi_out = np.sign(psi) * np.mod(np.abs(psi), 2 * math.pi)

    # restrict psi to [-pi,pi[
    if type(psi_out) is np.ndarray:
        psi_out[psi_out >= math.pi] -= 2 * math.pi
        psi_out[psi_out < -math.pi] += 2 * math.pi

    else:
        if psi_out >= math.pi:
            psi_out -= 2 * math.pi
        elif psi_out < -math.pi:
            psi_out += 2 * math.pi

    return psi_out

def calc_head_curv_an(coeffs_x: np.ndarray,
                      coeffs_y: np.ndarray,
                      ind_spls: np.ndarray,
                      t_spls: np.ndarray,) -> tuple:
    
    # calculate required derivatives
    x_d = coeffs_x[ind_spls, 1] \
          + 2 * coeffs_x[ind_spls, 2] * t_spls \
          + 3 * coeffs_x[ind_spls, 3] * np.power(t_spls, 2)

    y_d = coeffs_y[ind_spls, 1] \
          + 2 * coeffs_y[ind_spls, 2] * t_spls \
          + 3 * coeffs_y[ind_spls, 3] * np.power(t_spls, 2)

    # calculate heading psi (pi/2 must be substracted due to our convention that psi = 0 is north)
    psi = np.arctan2(y_d, x_d) - math.pi / 2
    psi = normalize_psi(psi)


    # calculate required derivatives
    x_dd = 2 * coeffs_x[ind_spls, 2] \
            + 6 * coeffs_x[ind_spls, 3] * t_spls

    y_dd = 2 * coeffs_y[ind_spls, 2] \
            + 6 * coeffs_y[ind_spls, 3] * t_spls

    # calculate curvature kappa
    kappa = (x_d * y_dd - y_d * x_dd) / np.power(np.power(x_d, 2) + np.power(y_d, 2), 1.5)

    return psi, kappa

def calc_ax_profile(vx_profile: np.ndarray,
                    el_lengths: np.ndarray,
                    eq_length_output: bool = False) -> np.ndarray:
 
     # calculate longitudinal acceleration profile array numerically: (v_end^2 - v_beg^2) / 2*s
    if eq_length_output:
        ax_profile = np.zeros(vx_profile.size)
        ax_profile[:-1] = (np.power(vx_profile[1:], 2) - np.power(vx_profile[:-1], 2)) / (2 * el_lengths)
    else:
        ax_profile = (np.power(vx_profile[1:], 2) - np.power(vx_profile[:-1], 2)) / (2 * el_lengths)

    return ax_profile

def calc_splines(path: np.ndarray,
                 el_lengths: np.ndarray = None,
                 psi_s: float = None,
                 psi_e: float = None,
                 use_dist_scaling: bool = True) -> tuple:
    
    # check if path is closed
    if np.all(np.isclose(path[0], path[-1])) and psi_s is None:
        closed = True
    else:
        closed = False

    # if distances between path coordinates are not provided but required, calculate euclidean distances as el_lengths
    if use_dist_scaling and el_lengths is None:
        el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))
    elif el_lengths is not None:
        el_lengths = np.copy(el_lengths)

    # if closed and use_dist_scaling active append element length in order to obtain overlapping elements for proper
    # scaling of the last element afterwards
    if use_dist_scaling and closed:
        el_lengths = np.append(el_lengths, el_lengths[0])

    # get number of splines
    no_splines = path.shape[0] - 1

    # calculate scaling factors between every pair of splines
    if use_dist_scaling:
        scaling = el_lengths[:-1] / el_lengths[1:]
    else:
        scaling = np.ones(no_splines - 1)

    # ------------------------------------------------------------------------------------------------------------------
    # DEFINE LINEAR EQUATION SYSTEM ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # M_{x,y} * a_{x,y} = b_{x,y}) with a_{x,y} being the desired spline param
    # *4 because of 4 parameters in cubic spline
    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))

    # create template for M array entries
    # row 1: beginning of current spline should be placed on current point (t = 0)
    # row 2: end of current spline should be placed on next point (t = 1)
    # row 3: heading at end of current spline should be equal to heading at beginning of next spline (t = 1 and t = 0)
    # row 4: curvature at end of current spline should be equal to curvature at beginning of next spline (t = 1 and
    #        t = 0)
    template_M = np.array(                          # current point               | next point              | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0

    for i in range(no_splines):
        j = i * 4

        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M

            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)

        else:
            # no curvature and heading bounds on last element (handled afterwards)
            M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],
                                     [1,  1,  1,  1]]

        b_x[j: j + 2] = [[path[i,     0]],
                         [path[i + 1, 0]]]
        b_y[j: j + 2] = [[path[i,     1]],
                         [path[i + 1, 1]]]

    # ------------------------------------------------------------------------------------------------------------------
    # SET BOUNDARY CONDITIONS FOR LAST AND FIRST POINT -----------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not closed:
        # if the path is unclosed we want to fix heading at the start and end point of the path (curvature cannot be
        # determined in this case) -> set heading boundary conditions

        # heading start point
        M[-2, 1] = 1  # heading start point (evaluated at t = 0)

        if el_lengths is None:
            el_length_s = 1.0
        else:
            el_length_s = el_lengths[0]

        b_x[-2] = math.cos(psi_s + math.pi / 2) * el_length_s
        b_y[-2] = math.sin(psi_s + math.pi / 2) * el_length_s

        # heading end point
        M[-1, -4:] = [0, 1, 2, 3]  # heading end point (evaluated at t = 1)

        if el_lengths is None:
            el_length_e = 1.0
        else:
            el_length_e = el_lengths[-1]

        b_x[-1] = math.cos(psi_e + math.pi / 2) * el_length_e
        b_y[-1] = math.sin(psi_e + math.pi / 2) * el_length_e

    else:
        # heading boundary condition (for a closed spline)
        M[-2, 1] = scaling[-1]
        M[-2, -3:] = [-1, -2, -3]
        # b_x[-2] = 0
        # b_y[-2] = 0

        # curvature boundary condition (for a closed spline)
        M[-1, 2] = 2 * math.pow(scaling[-1], 2)
        M[-1, -2:] = [-2, -6]
        # b_x[-1] = 0
        # b_y[-1] = 0

    x_les = np.squeeze(np.linalg.solve(M, b_x))  # squeeze removes single-dimensional entries
    y_les = np.squeeze(np.linalg.solve(M, b_y))

    # get coefficients of every piece into one row -> reshape
    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))

    # get normal vector (behind used here instead of ahead for consistency with other functions) (second coefficient of
    # cubic splines is relevant for the heading)
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

    # normalize normal vectors
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    normvec_normalized = np.expand_dims(norm_factors, axis=1) * normvec

    return coeffs_x, coeffs_y, M, normvec_normalized

def interp_splines(coeffs_x: np.ndarray,
                   coeffs_y: np.ndarray,
                   spline_lengths: np.ndarray = None,
                   incl_last_point: bool = False,
                   stepsize_approx: float = None,
                   stepnum_fixed: list = None) -> tuple:
    
    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE NUMBER OF INTERPOLATION POINTS AND ACCORDING DISTANCES -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if stepsize_approx is not None:
        # get the total distance up to the end of every spline (i.e. cumulated distances)
        if spline_lengths is None:
            spline_lengths = calc_spline_lengths(coeffs_x=coeffs_x,
                                                    coeffs_y=coeffs_y,
                                                    quickndirty=False)

        dists_cum = np.cumsum(spline_lengths)

        # calculate number of interpolation points and distances (+1 because last point is included at first)
        no_interp_points = math.ceil(dists_cum[-1] / stepsize_approx) + 1
        dists_interp = np.linspace(0.0, dists_cum[-1], no_interp_points)

    else:
        # get total number of points to be sampled (subtract overlapping points)
        no_interp_points = sum(stepnum_fixed) - (len(stepnum_fixed) - 1)
        dists_interp = None

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE INTERMEDIATE STEPS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create arrays to save the values
    path_interp = np.zeros((no_interp_points, 2))           # raceline coords (x, y) array
    spline_inds = np.zeros(no_interp_points, dtype=int)  # save the spline index to which a point belongs
    t_values = np.zeros(no_interp_points)                   # save t values

    if stepsize_approx is not None:

        # --------------------------------------------------------------------------------------------------------------
        # APPROX. EQUAL STEP SIZE ALONG PATH OF ADJACENT SPLINES -------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # loop through all the elements and create steps with stepsize_approx
        for i in range(no_interp_points - 1):
            # find the spline that hosts the current interpolation point
            j = np.argmax(dists_interp[i] < dists_cum)
            spline_inds[i] = j

            # get spline t value depending on the progress within the current element
            if j > 0:
                t_values[i] = (dists_interp[i] - dists_cum[j - 1]) / spline_lengths[j]
            else:
                if spline_lengths.ndim == 0:
                    t_values[i] = dists_interp[i] / spline_lengths
                else:
                    t_values[i] = dists_interp[i] / spline_lengths[0]

            # calculate coords
            path_interp[i, 0] = coeffs_x[j, 0] \
                                + coeffs_x[j, 1] * t_values[i]\
                                + coeffs_x[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_x[j, 3] * math.pow(t_values[i], 3)

            path_interp[i, 1] = coeffs_y[j, 0]\
                                + coeffs_y[j, 1] * t_values[i]\
                                + coeffs_y[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_y[j, 3] * math.pow(t_values[i], 3)

    else:

        # --------------------------------------------------------------------------------------------------------------
        # FIXED STEP SIZE FOR EVERY SPLINE SEGMENT ---------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        j = 0

        for i in range(len(stepnum_fixed)):
            # skip last point except for last segment
            if i < len(stepnum_fixed) - 1:
                t_values[j:(j + stepnum_fixed[i] - 1)] = np.linspace(0, 1, stepnum_fixed[i])[:-1]
                spline_inds[j:(j + stepnum_fixed[i] - 1)] = i
                j += stepnum_fixed[i] - 1

            else:
                t_values[j:(j + stepnum_fixed[i])] = np.linspace(0, 1, stepnum_fixed[i])
                spline_inds[j:(j + stepnum_fixed[i])] = i
                j += stepnum_fixed[i]

        t_set = np.column_stack((np.ones(no_interp_points), t_values, np.power(t_values, 2), np.power(t_values, 3)))

        # remove overlapping samples
        n_samples = np.array(stepnum_fixed)
        n_samples[:-1] -= 1

        path_interp[:, 0] = np.sum(np.multiply(np.repeat(coeffs_x, n_samples, axis=0), t_set), axis=1)
        path_interp[:, 1] = np.sum(np.multiply(np.repeat(coeffs_y, n_samples, axis=0), t_set), axis=1)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE LAST POINT IF REQUIRED (t = 1.0) -----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if incl_last_point:
        path_interp[-1, 0] = np.sum(coeffs_x[-1])
        path_interp[-1, 1] = np.sum(coeffs_y[-1])
        spline_inds[-1] = coeffs_x.shape[0] - 1
        t_values[-1] = 1.0

    else:
        path_interp = path_interp[:-1]
        spline_inds = spline_inds[:-1]
        t_values = t_values[:-1]

        if dists_interp is not None:
            dists_interp = dists_interp[:-1]

    # NOTE: dists_interp is None, when using a fixed step size
    return path_interp, spline_inds, t_values, dists_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
