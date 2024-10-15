import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.optim import lr_scheduler 

class ModelTrainer:
    def __init__(self, model, learning_rate, batch_size, device=None, early_stopping_patience=5,
                 weight_decay=0.0001, dropout_rate=0.1, clip_grad_norm=1.0, lr_scheduler_factor=0.1,
                 lr_scheduler_patience=3, mse_weight=0.5, mae_weight=0.5, pct_start=0.3, use_gpu=True, use_multi_gpu=False,
                 device_ids = 0):
        self.device = device if device is not None else torch.device("cpu" if torch.cpu.is_available() else "cuda")
        self.model = model.to(self.device)
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        '''self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_scheduler_factor, 
                                           patience=lr_scheduler_patience, verbose=True)'''
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        self.clip_grad_norm = clip_grad_norm
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

        self.use_gpu =use_gpu
        self.use_multi_gpu = use_multi_gpu
        self.device_ids = device_ids
        self.pct_start = pct_start

        if self.use_multi_gpu and self.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

    def train(self, train_dataset, val_dataset, num_epochs):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        best_val_loss = np.inf
        patience_counter = 0

        self.scheduler = lr_scheduler.OneCycleLR(optimizer = self.optimizer,
                                            steps_per_epoch = int(num_epochs/self.batch_size),
                                            pct_start = self.pct_start,
                                            epochs = num_epochs,
                                            max_lr = self.learning_rate)

        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()
            total_mse_loss = 0
            total_mae_loss = 0

            for batch in train_loader:
                x, y_true = [b.to(self.device) for b in batch]

                self.optimizer.zero_grad()
                y_pred = self.model(x)#, None, y_true, None)  # Passing None for x_mark and y_mark

                mse_loss = self.mse_criterion(y_pred, y_true)
                mae_loss = self.mae_criterion(y_pred, y_true)

                combined_loss = self.mse_weight * mse_loss + self.mae_weight * mae_loss
                combined_loss.backward()
                
                # Gradient clipping
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.optimizer.step()

                total_mse_loss += mse_loss.item()
                total_mae_loss += mae_loss.item()

            avg_mse_loss = np.average(total_mse_loss)
            avg_mae_loss = np.average(total_mae_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Train MSE Loss: {avg_mse_loss:.4f}, Train MAE Loss: {avg_mae_loss:.4f}")

            # Validation Phase
            val_mse_loss, val_mae_loss = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Val MSE Loss: {val_mse_loss:.4f}, Val MAE Loss: {val_mae_loss:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_mae_loss)

            # Early Stopping Check
            if val_mae_loss < best_val_loss:
                best_val_loss = val_mae_loss
                patience_counter = 0
                print(f"Validation loss improved. Saving model at epoch {epoch+1}.")
                #torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience counter: {patience_counter}/{self.early_stopping_patience}")

            # Stop early if patience exceeds the limit
            if patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered. Training halted.")
                break

    def validate(self, val_loader):
        self.model.eval()
        total_mse = 0
        total_mae = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y_true = [b.to(self.device) for b in batch]
                y_pred = self.model(x)#, None, y_true, None)  # Passing None for x_mark and y_mark

                mse = self.mse_criterion(y_pred, y_true)
                mae = self.mae_criterion(y_pred, y_true)

                total_mse += mse.item()
                total_mae += mae.item()

        avg_mse_loss = np.average(total_mse)
        avg_mae_loss = np.average(total_mae)
        return avg_mse_loss, avg_mae_loss

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
                y_pred = self.model(x)#, None, y_true, None)  # Passing None for x_mark and y_mark
                
                mse = self.mse_criterion(y_pred, y_true)
                mae = self.mae_criterion(y_pred, y_true)
                
                total_mse += mse.item()
                total_mae += mae.item()

                all_predictions.append(y_pred.cpu())
                all_true_values.append(y_true.cpu())
        
        avg_mse_loss = np.average(total_mse)
        avg_mae_loss = np.average(total_mae)
        
        print(f"Test MSE: {avg_mse_loss:.4f}")
        print(f"Test MAE: {avg_mae_loss:.4f}")

        # Concatenate all predictions and true values
        all_predictions = torch.cat(all_predictions, dim=0)
        all_true_values = torch.cat(all_true_values, dim=0)
        
        return all_predictions, all_true_values

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y_pred = self.model(x)#, None, None, None)  # Passing None for x_mark, y_true, and y_mark
        return y_pred.cpu()

    def set_dropout(self, dropout_rate):
        """
        Set dropout rate for all dropout layers in the model
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate