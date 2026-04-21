from train import evaluate
from dataset.station_dataset import StationDataset
import torch
from utils.metrics import HuberLoss, MAELoss, RMSELoss
from models.predictor import MultiViewPredictor

def test_evaluate():
    valid_dataset = StationDataset(mode="valid", cv_k=0)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=28,
        shuffle=False,
        num_workers=4,
    )
    
    loss_func_rmse = RMSELoss()
    loss_func_mae = MAELoss()
    
    net =net = MultiViewPredictor(num_station=10, output_size=1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    net.load_state_dict(torch.load("checkpoints/tmp_save_model.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    net.eval()
    test_huber, test_rmse, test_mae = evaluate(net, valid_loader, loss_func_rmse, loss_func_mae)
    print(f"Test Huber Loss: {test_huber:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
if __name__ == "__main__":
    test_evaluate() 