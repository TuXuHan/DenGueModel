import argparse
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config as cfg
from dataset.station_dataset import StationDataset
from models.predictor import MultiViewPredictor
from utils.logging_utils import setup_logging
from utils.metrics import HuberLoss, MAELoss, RMSELoss


def _assert_batch_shapes(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    if output.shape != target.shape:
        raise ValueError(f"Output/target shape mismatch: output={output.shape} target={target.shape}")
    if target.shape != mask.shape:
        raise ValueError(f"Target/mask shape mismatch: target={target.shape} mask={mask.shape}")


def _masked_real_sums(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    if torch.isnan(output).any() or torch.isnan(target).any():
        raise ValueError("NaN detected in output/target.")
    if (target < -1e-6).any():
        raise ValueError("Target has values < 0; expected log1p-space non-negative targets.")
    real_output = torch.expm1(output)
    real_target = torch.expm1(target)
    real_output = torch.clamp(real_output, min=0.0)
    diff = (real_output - real_target) * mask
    sse = (diff ** 2).sum()
    ae = torch.abs(diff).sum()
    valid = mask.sum()
    return sse, ae, valid


def evaluate(net, data_loader):
    net.eval()
    
    total_sse = 0.0
    total_ae = 0.0
    total_valid = 1e-9
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = [item.to(cfg.DEVICE) for item in batch]
            (
                ovi_target,
                target_mask,
                meo_unlabel,
                feature_unlabel,
                ovi_label,
                meo_label,
                feature_label_out,
                inv_dis_label,
                label_id_list,
                timestamp,
                _
            ) = batch
            h_t = torch.zeros(ovi_target.shape[0], 32, device=cfg.DEVICE)
            output = net(
                meo_unlabel,
                feature_unlabel,
                ovi_label,
                meo_label,
                feature_label_out,
                inv_dis_label,
                h_t,
                timestamp,
                label_id_list,
            )
            _assert_batch_shapes(output, ovi_target, target_mask)
            batch_sse, batch_ae, batch_valid = _masked_real_sums(output, ovi_target, target_mask)
            if batch_valid.item() == 0:
                continue
            total_sse += batch_sse.item()
            total_ae += batch_ae.item()
            total_valid += batch_valid.item()
            
    avg_mse = total_sse / total_valid
    avg_rmse = np.sqrt(avg_mse)
    avg_mae = total_ae / total_valid
    avg_huber = (avg_rmse + avg_mae) / 2
    return avg_huber, avg_rmse, avg_mae


def plot_history(history, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.plot(history)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train multiview predictor")
    parser.add_argument("--cv_k", type=int, default=0, help="k of cross validation")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loader")
    args = parser.parse_args()

    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    setup_logging(cfg.LOG_DIR, timestr)

    torch.manual_seed(cfg.RANDOM_SEED)

    nearest_col = cfg.nearest_columns(cfg.K_NEIGHBOR)
    near_dist_col = cfg.near_dist_columns(cfg.K_NEIGHBOR)

    train_dataset = StationDataset(mode="train", cv_k=args.cv_k)
    valid_dataset = StationDataset(mode="valid", cv_k=args.cv_k)
    test_dataset = StationDataset(mode="test", cv_k=args.cv_k)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE, pin_memory=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    net = MultiViewPredictor(num_station=cfg.K_NEIGHBOR, output_size=cfg.MODEL_OUTPUT_SIZE).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    (cfg.MODEL_DIR / timestr).mkdir(parents=True, exist_ok=True)
    loss_func_rmse = RMSELoss()
    loss_func_mae = MAELoss()

    # print("test performance before training...")
    # time.sleep(1)
    # test_huber_before, test_rmse_before, test_mae_before = evaluate(net, test_loader, loss_func_rmse, loss_func_mae)
    # print(f"Huber : {test_huber_before:.4f} RMSE : {test_rmse_before:.4f} MAE : {test_mae_before:.4f}")

    loss_history_tr_mae = []
    loss_history_val_mae = []
    loss_history_te_mae = []
    loss_history_tr_rmse = []
    loss_history_val_rmse = []
    loss_history_te_rmse = []
    loss_history_tr_huber = []
    loss_history_val_huber = []
    loss_history_te_huber = []

    patient_count = 0
    best_epoch = -1
    best_valid_loss = 1e9
    best_model_path = cfg.MODEL_DIR / timestr / "best_wts_my_model.pt"
    print("Using Device:", cfg.DEVICE)
    for epoch in range(cfg.MAX_EPOCH):
        try:
            net.train()
            running_loss_rmse = 0.0
            running_loss_mae = 0.0
            running_loss_mse = 0.0
            
            total_valid_samples = 1e-9
            skipped_batches = 0
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                batch = [item.to(cfg.DEVICE) for item in batch]
                (
                    ovi_target,
                    target_mask,
                    meo_unlabel,
                    feature_unlabel,
                    ovi_label,
                    meo_label,
                    feature_label_out,
                    inv_dis_label,
                    label_id_list,
                    timestamp,
                    _
                ) = batch
                optimizer.zero_grad()
                h_t = torch.zeros(ovi_target.shape[0], 32, device=cfg.DEVICE)
                output = net(
                    meo_unlabel,
                    feature_unlabel,
                    ovi_label,
                    meo_label,
                    feature_label_out,
                    inv_dis_label,
                    h_t,
                    timestamp,
                    label_id_list,
                )
                _assert_batch_shapes(output, ovi_target, target_mask)
                raw_mse = torch.nn.MSELoss(reduction="none")(output, ovi_target)    #log scale mse
                masked_mse = raw_mse * target_mask
                batch_valid = target_mask.sum()
                if batch_valid.item() == 0:
                    skipped_batches += 1
                    continue
                loss = masked_mse.sum() / (batch_valid + 1e-9)
                loss.backward()
                optimizer.step()

                # Metric Report for real scale
                with torch.no_grad():
                    batch_sse, batch_ae, batch_valid_count = _masked_real_sums(output, ovi_target, target_mask)
                    if batch_valid_count.item() == 0:
                        continue
                    running_loss_mse += batch_sse.item()
                    running_loss_mae += batch_ae.item()
                    total_valid_samples += batch_valid_count.item()

                    # Sanity: same-batch metrics should match evaluate() when in eval mode
                    if epoch == 0 and batch_idx == 0:
                        net.eval()
                        with torch.no_grad():
                            output_eval = net(
                                meo_unlabel,
                                feature_unlabel,
                                ovi_label,
                                meo_label,
                                feature_label_out,
                                inv_dis_label,
                                h_t,
                                timestamp,
                                label_id_list,
                            )
                            _assert_batch_shapes(output_eval, ovi_target, target_mask)
                            eval_sse, eval_ae, eval_valid = _masked_real_sums(output_eval, ovi_target, target_mask)
                            if eval_valid.item() != 0:
                                inline_mse = eval_sse / eval_valid
                                inline_rmse = torch.sqrt(inline_mse)
                                inline_mae = eval_ae / eval_valid
                                inline_huber = (inline_rmse + inline_mae) / 2
                                eval_huber, eval_rmse, eval_mae = evaluate(net, [[item.detach().cpu() for item in batch]])
                                eval_huber_t = torch.tensor(eval_huber, device=inline_huber.device, dtype=inline_huber.dtype)
                                eval_rmse_t = torch.tensor(eval_rmse, device=inline_rmse.device, dtype=inline_rmse.dtype)
                                eval_mae_t = torch.tensor(eval_mae, device=inline_mae.device, dtype=inline_mae.dtype)
                                if not (
                                    torch.isclose(inline_huber, eval_huber_t, rtol=1e-4, atol=1e-4)
                                    and torch.isclose(inline_rmse, eval_rmse_t, rtol=1e-4, atol=1e-4)
                                    and torch.isclose(inline_mae, eval_mae_t, rtol=1e-4, atol=1e-4)
                                ):
                                    raise ValueError("Sanity check failed: evaluate() mismatch on same batch.")
                        net.train()

            avg_mse = running_loss_mse / total_valid_samples
            avg_rmse = np.sqrt(avg_mse)
            avg_mae = running_loss_mae / total_valid_samples
            avg_huber = (avg_rmse + avg_mae) / 2
            if skipped_batches > 0:
                print(f"[Epoch {epoch:2d}] Skipped {skipped_batches} batches with zero valid targets.")
            torch.save(net.state_dict(), cfg.MODEL_DIR / timestr / "tmp_save_model.pt")
            print(f"[Epoch {epoch:2d}] Training Huber : {avg_huber:.4f} Training RMSE : {avg_rmse:.4f} Training MAE : {avg_mae:.4f} Training MSE : {avg_mse:.4f}")
            loss_history_tr_mae.append(avg_mae)
            loss_history_tr_rmse.append(avg_rmse)
            loss_history_tr_huber.append(avg_huber)

            print("Start validation...")
            time.sleep(1)
            valid_huber, valid_rmse, valid_mae = evaluate(net, valid_loader)
            print(f"[Epoch {epoch:2d}] Validation Huber : {valid_huber:.4f} Validation RMSE : {valid_rmse:.4f} Validation MAE : {valid_mae:.4f}")
            loss_history_val_mae.append(valid_mae)
            loss_history_val_rmse.append(valid_rmse)
            loss_history_val_huber.append(valid_huber)

            if valid_huber < best_valid_loss:
                best_valid_loss = valid_huber
                patient_count = 0
                torch.save(net.state_dict(), best_model_path)
                print(f"Save model at epoch {epoch}")
                best_epoch = epoch
            else:
                patient_count += 1
                if patient_count == cfg.PATIENCE:
                    break
            time.sleep(1)

            print("Start testing...")
            time.sleep(1)
            test_huber, test_rmse, test_mae = evaluate(net, test_loader)
            print(f"[Epoch {epoch:2d}] Testing Huber : {test_huber:.4f} Testing RMSE : {test_rmse:.4f} Testing MAE : {test_mae:.4f}")
            loss_history_te_mae.append(test_mae)
            loss_history_te_rmse.append(test_rmse)
            loss_history_te_huber.append(test_huber)
            time.sleep(1)

        except Exception as e:  # noqa: F841
            torch.save(net.state_dict(), cfg.MODEL_DIR / timestr / "tmp_save_model.pt")
            print("Exception in epoch loop:", repr(e))
            raise

    print("loss history: ")
    print("train RMSE", loss_history_tr_rmse)
    print("valid RMSE", loss_history_val_rmse)
    print("test RMSE", loss_history_te_rmse)
    print("train MAE", loss_history_tr_mae)
    print("valid MAE", loss_history_val_mae)
    print("test MAE", loss_history_te_mae)
    print("train Huber", loss_history_tr_huber)
    print("valid Huber", loss_history_val_huber)
    print("test Huber", loss_history_te_huber)

    if best_epoch >= 0:
        print("best loss: ")
        print(
            f"[Epoch {best_epoch:2d}] Training Huber : {loss_history_tr_huber[best_epoch]:.4f} Training RMSE : {loss_history_tr_rmse[best_epoch]:.4f} Training MAE : {loss_history_tr_mae[best_epoch]:.4f}"
        )
        print(
            f"[Epoch {best_epoch:2d}] Validation Huber : {loss_history_val_huber[best_epoch]:.4f} Validation RMSE : {loss_history_val_rmse[best_epoch]:.4f} Validation MAE : {loss_history_val_mae[best_epoch]:.4f}"
        )
        print(
            f"[Epoch {best_epoch:2d}] Testing Huber : {loss_history_te_huber[best_epoch]:.4f} Testing RMSE : {loss_history_te_rmse[best_epoch]:.4f} Testing MAE : {loss_history_te_mae[best_epoch]:.4f}"
        )

    fig_dir = cfg.FIG_DIR / timestr
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_history(loss_history_tr_rmse, fig_dir / "rmse_train.png")
    plot_history(loss_history_val_rmse, fig_dir / "rmse_valid.png")
    plot_history(loss_history_te_rmse, fig_dir / "rmse_test.png")
    plot_history(loss_history_tr_mae, fig_dir / "mae_train.png")
    plot_history(loss_history_val_mae, fig_dir / "mae_valid.png")
    plot_history(loss_history_te_mae, fig_dir / "mae_test.png")
    plot_history(loss_history_tr_huber, fig_dir / "huber_train.png")
    plot_history(loss_history_val_huber, fig_dir / "huber_valid.png")
    plot_history(loss_history_te_huber, fig_dir / "huber_test.png")


if __name__ == "__main__":
    main()
