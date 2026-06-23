# SC_FTMixer_SDE_Unified — Standalone training entry
import os, io, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')

import argparse, random, numpy as np, torch, time

from experiments.exp_long_term_forecasting_sde import Exp_Long_Term_Forecast as Exp_SDE
from experiments.exp_cemp_regression import Exp_CEMP_Regression


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SC_FTMixer_SDE_Unified')

    # Basic
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, default='SC_FTMixer_SDE_Unified')
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=2021)

    # Data
    parser.add_argument('--data', type=str, required=True, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='data.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', type=str, default='timeF')

    # Sequence
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # Model structure
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--inverse', action='store_true', default=False)
    parser.add_argument('--visualize', type=int, default=0,help="Whether to draw figures, 1=enable visualization, 0=disable visualization")

    # Training
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--lradj', type=str, default='cosine_warmup')
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # SDE
    parser.add_argument('--use_sde', type=int, default=1)
    parser.add_argument('--sde_cycle_len', type=int, default=24)
    parser.add_argument('--sde_hidden', type=int, default=192)
    parser.add_argument('--sde_rec_weight', type=float, default=0.25)
    parser.add_argument('--sde_spectral_weight', type=float, default=0.75)
    parser.add_argument('--sde_phase_mode', type=str, default='auto')
    parser.add_argument('--sde_slots_per_hour', type=int, default=1)
    parser.add_argument('--sde_calendar_gate_init', type=float, default=2.0)
    parser.add_argument('--use_global_sde', type=int, default=1)
    parser.add_argument('--use_calendar_sde', type=int, default=1)
    parser.add_argument('--use_dynamic_filter', type=int, default=1)
    parser.add_argument('--fix_calendar_gate', type=int, default=0)

    # Mixed loss
    parser.add_argument('--mae_weight', type=float, default=0.0)
    parser.add_argument('--diff_loss_weight', type=float, default=0.0)

    args = parser.parse_args()
    base_seed = args.seed
    random.seed(base_seed)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_SDE
    elif args.task_name == 'cemp_regression':
        Exp = Exp_CEMP_Regression
    else:
        raise ValueError("task_name must be long_term_forecast or cemp_regression")

    if args.is_training:
        all_metrics = []
        for ii in range(args.itr):
            cur_seed = base_seed + ii
            random.seed(cur_seed)
            torch.manual_seed(cur_seed)
            np.random.seed(cur_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cur_seed)
                torch.cuda.manual_seed_all(cur_seed)

            setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_{}_{}_dp{}'.format(
                args.task_name, args.model_id, args.model,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers,
                args.des, ii, args.dropout)

            print(f'>>>>>>> seed for itr {ii}: {cur_seed}')
            exp = Exp(args)
            if args.use_gpu:
                torch.cuda.reset_peak_memory_stats()
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

            if args.task_name == 'long_term_forecast':
                mse, mae = exp.test(setting)
                all_metrics.append((mse, mae))
                print(f'[Run {ii}] mse={mse:.7f}, mae={mae:.7f}')
            else:
                exp.test(setting)
            if args.use_gpu:
                print(
                    "REVIEW_PEAK_MEMORY_MIB="
                    f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.3f}"
                )
            torch.cuda.empty_cache()

        if args.task_name == 'long_term_forecast' and args.itr > 1 and len(all_metrics) > 0:
            mses = np.array([m[0] for m in all_metrics], dtype=np.float64)
            maes = np.array([m[1] for m in all_metrics], dtype=np.float64)
            print("\n" + "=" * 60)
            print("Multi-run summary:")
            for i, (mse, mae) in enumerate(all_metrics):
                print(f"  run {i}: mse={mse:.7f}, mae={mae:.7f}")
            print(f"  mean : mse={mses.mean():.7f}, mae={maes.mean():.7f}")
            print(f"  std  : mse={mses.std(ddof=0):.7f}, mae={maes.std(ddof=0):.7f}")
            print("=" * 60 + "\n")

            log_dir = "./results/log_mse_mae"
            os.makedirs(os.path.join(log_dir, "All"), exist_ok=True)
            result_filename = os.path.join(log_dir, "All", f"{args.model}_ALL_forecast_RESULTS.txt")
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(result_filename, 'a') as f:
                f.write("===============================================\n")
                f.write(f"Model: {args.model} | Time: {current_time}\n")
                f.write(f"Args: {args}\n")
                f.write(f"Multi-run summary (itr={args.itr})\n")
                for i, (mse, mae) in enumerate(all_metrics):
                    f.write(f"run{i}: mse={mse}, mae={mae}\n")
                f.write(f"mean_mse:{mses.mean()}, mean_mae:{maes.mean()}\n")
                f.write(f"std_mse:{mses.std(ddof=0)}, std_mae:{maes.std(ddof=0)}\n")
                f.write("===============================================\n\n")
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_{}_{}_dp{}'.format(
            args.task_name, args.model_id, args.model,
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers,
            args.des, ii, args.dropout)
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if args.task_name == 'long_term_forecast':
            exp.test(setting, test=1)
        else:
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
