import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from dataset.station_dataset import StationDataset
from models.predictor import MultiViewPredictor

def _to_device(x):
    return x.to(cfg.DEVICE) if torch.is_tensor(x) else x


def _write_csv(path: Path, preds: torch.Tensor, ids: torch.Tensor, times: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds_np = preds.cpu().numpy()
    ids_np = ids.cpu().numpy()
    times_np = times.cpu().numpy()
    header = ["id", "timestamp"] + [f"pred_{i}" for i in range(preds_np.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(preds_np.shape[0]):
            row = [int(ids_np[i]), int(times_np[i])] + preds_np[i].tolist()
            writer.writerow(row)


def inference(net, data_loader, dry_run: bool = False):
    net.eval()
    preds = []
    ids = []
    times = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            batch = [_to_device(item) for item in batch]
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
                unlabel_id,
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
            real_output = torch.expm1(output)
            real_output = torch.clamp(real_output, min=0.0)
            preds.append(real_output.detach().cpu())
            ids.append(unlabel_id.detach().cpu())
            times.append(timestamp.detach().cpu())
            if dry_run:
                log_min = output.min().item()
                log_max = output.max().item()
                log_mean = output.mean().item()
                real_min = real_output.min().item()
                real_max = real_output.max().item()
                real_mean = real_output.mean().item()
                print(
                    "Dry-run batch",
                    batch_idx,
                    "output",
                    tuple(output.shape),
                    "target_mask",
                    tuple(target_mask.shape),
                )
                print(
                    f"  log-space: min={log_min:.6f} max={log_max:.6f} mean={log_mean:.6f}"
                )
                print(
                    f"  real-space: min={real_min:.6f} max={real_max:.6f} mean={real_mean:.6f}"
                )
                print(f"  accumulated samples: {sum(x.shape[0] for x in preds)}")
                if batch_idx >= 1:
                    break
    if not preds:
        return None, None, None
    return torch.cat(preds, dim=0), torch.cat(ids, dim=0), torch.cat(times, dim=0)

def main():
    parser = argparse.ArgumentParser(description="MVGFRNN Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--output", type=str, required=False, help="Path to save outputs (.pt or .csv)")
    parser.add_argument("--dry_run", action="store_true", help="Run only 2 batches and print stats")
    args = parser.parse_args()
    dataset = StationDataset(mode="infer", cv_k=0)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=False)

    net = MultiViewPredictor(num_station=cfg.K_NEIGHBOR, output_size=cfg.MODEL_OUTPUT_SIZE).to(cfg.DEVICE)
    net.load_state_dict(torch.load(args.model, map_location=cfg.DEVICE))

    preds, ids, times = inference(net, dataloader, dry_run=args.dry_run)
    if preds is None:
        print("Inference output: <empty>")
        return
    print("Inference output:", preds)
    if args.output:
        out_path = Path(args.output)
        if out_path.suffix in {".pt", ".pth"}:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"preds": preds, "ids": ids, "times": times}, out_path)
        elif out_path.suffix == ".csv":
            _write_csv(out_path, preds, ids, times)
        else:
            raise ValueError("Unsupported output format. Use .pt, .pth, or .csv.")

if __name__ == "__main__":
    main()
