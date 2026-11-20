import os
import numpy as np
import argparse
from estimators import laplace, moments, PBI
from utils import load_btcs, errors, synthetic, dimensional_reduction


def load_measured_btcs(btcs_csv_path, Nt):
    return load_btcs.data(btcs_csv_path, complete_zeros=True, min_dimless_res=1 / Nt)


def generate_synthetic_data(args):
    # Load BTCs
    n_btcs, ts_m, btcs_m, xs_m, names_list, T_ = load_measured_btcs(args.btcs_csv_path, args.Nt)
    t_ = np.linspace(0, T_, int(args.Nt * T_) + 1)[1:]

    # Estimate parameters using Laplace and Moments
    params_lap = np.array([laplace.ADEMF_1D(ts_m[i], btcs_m[i], xs_m[i], args.memory_func, args.bound_cond)
                           for i in range(n_btcs)])
    params_mom = np.array([moments.TSM(ts_m[i], btcs_m[i], xs_m[i], args.bound_cond)
                           for i in range(n_btcs)])

    # Compute errors
    errors_lap = np.array([errors.compute_errors(ts_m[i], btcs_m[i], xs_m[i], params_lap[i], args.memory_func, args.bound_cond)
                           for i in range(n_btcs)])
    errors_mom = np.array([errors.compute_errors(ts_m[i], btcs_m[i], xs_m[i], params_mom[i], args.memory_func, args.bound_cond)
                           for i in range(n_btcs)])

    # Select best estimates between Laplace and Moments
    best_params = params_lap.copy()
    if args.memory_func == 'first order':
        better_mom = errors_mom[:, 0] < errors_lap[:, 0]
        best_params[better_mom] = params_mom[better_mom]

    # Create log-normal distribution for parameter generation
    mean_log_params, cov_log_params, _ = synthetic.generate_dist(best_params[:, 1:])
    cov_log_params *= 2.0  # Widen prior

    # Generate synthetic dataset
    params_synth, btcs_synth = synthetic.generate(args.seed, args.n_synth, t_, args.memory_func,
                                                  args.bound_cond, mean_log_params, cov_log_params)

    # KL decomposition
    btcs_mean, phis, lambdas, Zs = dimensional_reduction.KL_decomposition(btcs_synth, args.n_lmbds, 1 / args.Nt)

    # Save
    os.makedirs(args.data_dir, exist_ok=True)
    def save_array(name, arr):
        path = f"{args.data_dir}/{name}_{args.n_synth}_{args.memory_func}_{args.bound_cond}_{args.seed}_{T_}.npy"
        np.save(path, arr)

    save_array("params_synth", params_synth)
    save_array("btcs_mean", btcs_mean)
    save_array("phis", phis)
    save_array("lambdas", lambdas)
    save_array("Zs", Zs)

    print("Synthetic dataset created and saved.")


def estimate_parameters_workflow(args):
    n_btcs, ts_m, btcs_m, xs_m, names_list, _ = load_measured_btcs(args.btcs_csv_path, args.Nt)
    t_ = np.linspace(0, args.T_synth, int(args.Nt * args.T_synth) + 1)[1:]

    # Load synthetic training data
    def load(name):
        return np.load(f"{args.data_dir}/{name}_{args.n_synth}_{args.memory_func}_{args.bound_cond}_{args.seed}_{args.T_synth}.npy")

    params_synth = load("params_synth")
    btcs_mean = load("btcs_mean")
    phis = load("phis")
    lambdas = load("lambdas")
    Zs = load("Zs")

    # KL coefficient estimation
    v_epsilons = np.linspace(args.v_range[0], args.v_range[1], args.n_vs)
    KL_out = [PBI.KL_coeffs(ts_m[i], btcs_m[i], xs_m[i], t_, btcs_mean, phis, lambdas, v_epsilons)
              for i in range(n_btcs)]
    Zs_meas_v = np.array([out[0] for out in KL_out])
    v_ests = np.array([out[1] for out in KL_out])

    # Parameter estimation
    params_NNI = np.array([
        PBI.estimate_params_NNI(params_synth, Zs, v_ests[i], Zs_meas_v[i], reg_v=0)
        for i in range(n_btcs)
    ])
    params_PBI = np.array([
        PBI.estimate_params_PBI(params_synth, Zs, v_ests[i], Zs_meas_v[i], 3, reg_v=0)
        for i in range(n_btcs)
    ])

    # Error evaluation
    errors_NNI = np.array([
        errors.compute_errors(ts_m[i], btcs_m[i], xs_m[i], params_NNI[i], args.memory_func, args.bound_cond)
        for i in range(n_btcs)
    ])
    errors_PBI = np.array([
        errors.compute_errors(ts_m[i], btcs_m[i], xs_m[i], params_PBI[i], args.memory_func, args.bound_cond)
        for i in range(n_btcs)
    ])
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save parameters
    if args.memory_func == 'first order':
        np.savetxt(f"{args.output_dir}/params_NNI.csv", params_NNI, delimiter=",", header="v,Pe,beta*k~_f,k~_r", comments='')
        np.savetxt(f"{args.output_dir}/params_PBI.csv", params_PBI, delimiter=",", header="v,Pe,beta*k~_f,k~_r", comments='')
    if args.memory_func == 'power law':
        np.savetxt(f"{args.output_dir}/params_NNI.csv", params_NNI, delimiter=",", header="v,Pe,beta*alpha~,1-gamma", comments='')
        np.savetxt(f"{args.output_dir}/params_PBI.csv", params_PBI, delimiter=",", header="v,Pe,beta*alpha~,1-gamma", comments='')
    
    # Save errors
    np.savetxt(f"{args.output_dir}/errors_NNI.csv", errors_NNI, delimiter=",", header="RMSE,KLdiv", comments='')
    np.savetxt(f"{args.output_dir}/errors_PBI.csv", errors_PBI, delimiter=",", header="RMSE,KLdiv", comments='')
    print("Estimation results saved.")


def main():
    parser = argparse.ArgumentParser(description="Run synthetic generation or parameter estimation.")
    parser.add_argument('--mode', choices=['generate_synthetic', 'estimate_parameters'], required=True)

    # Common
    parser.add_argument('--btcs_csv_path', type=str, default='data/antietam_creek_tracer_data.csv')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--Nt', type=int, default=150)
    parser.add_argument('--n_synth', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--memory_func', type=str, default='first order')
    parser.add_argument('--bound_cond', type=str, default='infinite')

    # Estimation-specific
    parser.add_argument('--v_range', type=list, default=[0.9,1.5])
    parser.add_argument('--n_vs', type=int, default=121)
    parser.add_argument('--T_synth', type=int, default=24)
    parser.add_argument('--output_dir', type=str, default='output')

    # Generation-specific
    parser.add_argument('--n_lmbds', type=int, default=35)

    args = parser.parse_args()

    if args.mode == 'generate_synthetic':
        generate_synthetic_data(args)
    elif args.mode == 'estimate_parameters':
        estimate_parameters_workflow(args)


if __name__ == '__main__':
    main()
