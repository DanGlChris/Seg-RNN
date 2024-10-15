import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class ModelTrainer:
    def __init__(self, model, learning_rate, batch_size, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size

    def train(self, train_dataset, num_epochs):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_mse_loss = 0
            total_mae_loss = 0
            
            for batch in train_loader:
                x, y_true = [b.to(self.device) for b in batch]
                
                self.optimizer.zero_grad()
                y_pred = self.model(x, None, y_true, None)  # Passing None for x_mark and y_mark
                
                mse_loss = self.mse_criterion(y_pred, y_true)
                mae_loss = self.mae_criterion(y_pred, y_true)
                
                # You can adjust the weights of MSE and MAE if needed
                combined_loss = mse_loss + mae_loss
                
                combined_loss.backward()
                self.optimizer.step()
                
                total_mse_loss += mse_loss.item()
                total_mae_loss += mae_loss.item()
            
            avg_mse_loss = total_mse_loss / len(train_loader)
            avg_mae_loss = total_mae_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, MSE Loss: {avg_mse_loss:.4f}, MAE Loss: {avg_mae_loss:.4f}")

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.model.eval()
        total_mse = 0
        total_mae = 0
        all_predictions = []
        all_true_values = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y_true = [b.to(self.device) for b in batch]
                y_pred = self.model(x, None, y_true, None)  # Passing None for x_mark and y_mark
                
                mse = nn.MSELoss()(y_pred, y_true)
                mae = nn.L1Loss()(y_pred, y_true)
                
                total_mse += mse.item()
                total_mae += mae.item()

                all_predictions.append(y_pred.cpu())
                all_true_values.append(y_true.cpu())
        
        avg_mse = total_mse / len(test_loader)
        avg_mae = total_mae / len(test_loader)
        
        print(f"Test MSE: {avg_mse:.4f}")
        print(f"Test MAE: {avg_mae:.4f}")

        # Concatenate all predictions and true values
        all_predictions = torch.cat(all_predictions, dim=0)
        all_true_values = torch.cat(all_true_values, dim=0)
        
        return all_predictions, all_true_values

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y_pred = self.model(x)  # Passing None for x_mark, y_true, and y_mark
        return y_pred.cpu()