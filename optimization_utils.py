import numpy as np
import math
import quadprog
# import cvxopt
import time

def opt_min_curv(reftrack: np.ndarray,
                 normvectors: np.ndarray,
                 A: np.ndarray,
                 kappa_bound: float,
                 w_veh: float,
                 closed: bool = True) -> tuple:

    no_points = reftrack.shape[0]

    no_splines = no_points
    if not closed:
        no_splines -= 1

    # check inputs
    if no_points != normvectors.shape[0]:
        raise RuntimeError("Array size of reftrack should be the same as normvectors!")

    if (no_points * 4 != A.shape[0] and closed) or (no_splines * 4 != A.shape[0] and not closed)\
            or A.shape[0] != A.shape[1]:
        raise RuntimeError("Spline equation system matrix A has wrong dimensions!")

    # create extraction matrix -> only b_i coefficients of the solved linear equation system are needed for gradient
    # information
    A_ex_b = np.zeros((no_points, no_splines * 4), dtype=int)

    for i in range(no_splines):
        A_ex_b[i, i * 4 + 1] = 1    # 1 * b_ix = E_x * x

    # coefficients for end of spline (t = 1)
    if not closed:
        A_ex_b[-1, -4:] = np.array([0, 1, 2, 3])

    # create extraction matrix -> only c_i coefficients of the solved linear equation system are needed for curvature
    # information
    A_ex_c = np.zeros((no_points, no_splines * 4), dtype=int)

    for i in range(no_splines):
        A_ex_c[i, i * 4 + 2] = 2    # 2 * c_ix = D_x * x

    # coefficients for end of spline (t = 1)
    if not closed:
        A_ex_c[-1, -4:] = np.array([0, 0, 2, 6])

    # invert matrix A resulting from the spline setup linear equation system and apply extraction matrix
    A_inv = np.linalg.inv(A)
    T_c = np.matmul(A_ex_c, A_inv)

    # set up M_x and M_y matrices including the gradient information, i.e. bring normal vectors into matrix form
    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))

    for i in range(no_splines):
        j = i * 4

        if i < no_points - 1:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, i + 1] = normvectors[i + 1, 0]

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, i + 1] = normvectors[i + 1, 1]
        else:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, 0] = normvectors[0, 0]  # close spline

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, 0] = normvectors[0, 1]

    # set up q_x and q_y matrices including the point coordinate information
    q_x = np.zeros((no_splines * 4, 1))
    q_y = np.zeros((no_splines * 4, 1))

    for i in range(no_splines):
        j = i * 4

        if i < no_points - 1:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[i + 1, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[i + 1, 1]
        else:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[0, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[0, 1]

    # set up P_xx, P_xy, P_yy matrices
    x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x)
    y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)

    x_prime_sq = np.power(x_prime, 2)
    y_prime_sq = np.power(y_prime, 2)
    x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)

    curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)                   # calculate curvature denominator
    curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                          where=curv_den != 0)                          # divide where not zero (diag elements)
    curv_part_sq = np.power(curv_part, 2)

    P_xx = np.matmul(curv_part_sq, y_prime_sq)
    P_yy = np.matmul(curv_part_sq, x_prime_sq)
    P_xy = np.matmul(curv_part_sq, x_prime_y_prime)


    ##################### SOLVING #####################

    T_nx = np.matmul(T_c, M_x)
    T_ny = np.matmul(T_c, M_y)

    H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
    H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
    H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
    H = H_x + H_xy + H_y
    H = (H + H.T) / 2   # make H symmetric

    f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
    f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
           + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
    f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
    f = f_x + f_xy + f_y
    f = np.squeeze(f)   # remove non-singleton dimensions


    #################### KAPPA #####################

    Q_x = np.matmul(curv_part, y_prime)
    Q_y = np.matmul(curv_part, x_prime)

    # this part is multiplied by alpha within the optimization (variable part)
    E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)

    # original curvature part (static part)
    k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

    con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
    con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)  # multiplied by -1 as only LE conditions are poss.
    con_stack = np.append(con_ge, con_le)

    
    ####################### QUADRATIC PRORAMMING ########################

    # calculate allowed deviation from refline
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2

    # check that there is space remaining between left and right maximum deviation (both can be negative as well!)
    if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
        raise RuntimeError("Problem not solvable, track might be too small to run with current safety distance!")

    # consider value boundaries (-dev_max_left <= alpha <= dev_max_right)
    G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
    h = np.append(dev_max_right, dev_max_left)
    h = np.append(h, con_stack)

    # save start time
    t_start = time.perf_counter()

    # solve problem (quadprog) -----------------------------------------------------------------------------------------
    alpha_mincurv = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

    
    ################## CURVATURE ERROR ##########################

    # calculate curvature once based on original linearization and once based on a new linearization around the solution
    q_x_tmp = q_x + np.matmul(M_x, np.expand_dims(alpha_mincurv, 1))
    q_y_tmp = q_y + np.matmul(M_y, np.expand_dims(alpha_mincurv, 1))

    x_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x_tmp)
    y_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y_tmp)

    x_prime_prime = np.squeeze(np.matmul(T_c, q_x) + np.matmul(T_nx, np.expand_dims(alpha_mincurv, 1)))
    y_prime_prime = np.squeeze(np.matmul(T_c, q_y) + np.matmul(T_ny, np.expand_dims(alpha_mincurv, 1)))

    curv_orig_lin = np.zeros(no_points)
    curv_sol_lin = np.zeros(no_points)

    for i in range(no_points):
        curv_orig_lin[i] = (x_prime[i, i] * y_prime_prime[i] - y_prime[i, i] * x_prime_prime[i]) \
                          / math.pow(math.pow(x_prime[i, i], 2) + math.pow(y_prime[i, i], 2), 1.5)
        curv_sol_lin[i] = (x_prime_tmp[i, i] * y_prime_prime[i] - y_prime_tmp[i, i] * x_prime_prime[i]) \
                           / math.pow(math.pow(x_prime_tmp[i, i], 2) + math.pow(y_prime_tmp[i, i], 2), 1.5)

    # calculate maximum curvature error
    curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))

    return alpha_mincurv, curv_error_max

def opt_shortest_path(reftrack: np.ndarray,
                      normvectors: np.ndarray,
                      w_veh: float,
                      print_debug: bool = False) -> np.ndarray:

    no_points = reftrack.shape[0]

    # check inputs
    if no_points != normvectors.shape[0]:
        raise RuntimeError("Array size of reftrack should be the same as normvectors!")

    H = np.zeros((no_points, no_points))
    f = np.zeros(no_points)

    for i in range(no_points):
        if i < no_points - 1:
            H[i, i] += 2 * (math.pow(normvectors[i, 0], 2) + math.pow(normvectors[i, 1], 2))
            H[i, i + 1] = 0.5 * 2 * (-2 * normvectors[i, 0] * normvectors[i + 1, 0]
                                     - 2 * normvectors[i, 1] * normvectors[i + 1, 1])
            H[i + 1, i] = H[i, i + 1]
            H[i + 1, i + 1] = 2 * (math.pow(normvectors[i + 1, 0], 2) + math.pow(normvectors[i + 1, 1], 2))

            f[i] += 2 * normvectors[i, 0] * reftrack[i, 0] - 2 * normvectors[i, 0] * reftrack[i + 1, 0] \
                    + 2 * normvectors[i, 1] * reftrack[i, 1] - 2 * normvectors[i, 1] * reftrack[i + 1, 1]
            f[i + 1] = -2 * normvectors[i + 1, 0] * reftrack[i, 0] \
                       - 2 * normvectors[i + 1, 1] * reftrack[i, 1] \
                       + 2 * normvectors[i + 1, 0] * reftrack[i + 1, 0] \
                       + 2 * normvectors[i + 1, 1] * reftrack[i + 1, 1]

        else:
            H[i, i] += 2 * (math.pow(normvectors[i, 0], 2) + math.pow(normvectors[i, 1], 2))
            H[i, 0] = 0.5 * 2 * (-2 * normvectors[i, 0] * normvectors[0, 0] - 2 * normvectors[i, 1] * normvectors[0, 1])
            H[0, i] = H[i, 0]
            H[0, 0] += 2 * (math.pow(normvectors[0, 0], 2) + math.pow(normvectors[0, 1], 2))

            f[i] += 2 * normvectors[i, 0] * reftrack[i, 0] - 2 * normvectors[i, 0] * reftrack[0, 0] \
                    + 2 * normvectors[i, 1] * reftrack[i, 1] - 2 * normvectors[i, 1] * reftrack[0, 1]
            f[0] += -2 * normvectors[0, 0] * reftrack[i, 0] - 2 * normvectors[0, 1] * reftrack[i, 1] \
                    + 2 * normvectors[0, 0] * reftrack[0, 0] + 2 * normvectors[0, 1] * reftrack[0, 1]

    ############## OPTIMIZATION ###############

    # calculate allowed deviation from refline
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2

    # set minimum deviation to zero
    dev_max_right[dev_max_right < 0.001] = 0.001
    dev_max_left[dev_max_left < 0.001] = 0.001

    # consider value boundaries (-dev_max <= alpha <= dev_max)
    G = np.vstack((np.eye(no_points), -np.eye(no_points)))
    h = np.ones(2 * no_points) * np.append(dev_max_right, dev_max_left)

    # save start time
    t_start = time.perf_counter()

    # solve problem
    alpha_shpath = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

    print("Solver runtime opt_shortest_path: " + "{:.3f}".format(time.perf_counter() - t_start) + "s")

    return alpha_shpath

# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass