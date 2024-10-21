import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.optim import lr_scheduler 

from adjacent_matrix_norm import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, calculate_symmetric_message_passing_adj, calculate_transition_matrix

class MASELoss(nn.Module):
    def __init__(self, naive_period=1, epsilon=1e-8):
        super().__init__()
        self.naive_period = naive_period
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Calculate MAE of the prediction
        mae_pred = F.l1_loss(y_pred, y_true, reduction='mean')
        
        # Calculate the naive forecast error
        naive_forecast = y_true[self.naive_period:]
        naive_error = y_true[self.naive_period:] - y_true[:-self.naive_period]
        
        # Calculate MAE of the naive forecast
        mae_naive = torch.mean(torch.abs(naive_error))
        
        # Calculate MASE
        mase = mae_pred / (mae_naive + self.epsilon)
        return mase
    
class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Calculate the absolute percentage error
        percentage_error = torch.abs((y_true - y_pred) / (y_true + self.epsilon))
        
        # Calculate MAPE
        mape = torch.mean(percentage_error) * 100
        return mape

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, yhat, y):
        return torch.mean((yhat - y) ** 2)
    
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 0.5 * F.mse_loss(x, y) + 0.5 * F.l1_loss(x, y)
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
                                            steps_per_epoch = 5,
                                            pct_start = self.pct_start,
                                            epochs = num_epochs,
                                            max_lr = self.learning_rate)

        self.criterion = MSELoss()

        for epoch in range(num_epochs):
            # Training Phase
            self.model.train()
            total_mse_loss = 0
            total_mae_loss = 0            
            train_loss = []

            for batch in train_loader:
                x, y_true = [b.to(self.device) for b in batch]

                self.optimizer.zero_grad()
                y_pred = self.model(x.unsqueeze(-1))#, None, y_true, None)  # Passing None for x_mark and y_mark

                #mse_loss = self.mse_criterion(y_pred, y_true)
                #mae_loss = self.mae_criterion(y_pred, y_true)
                loss = self.criterion(y_pred, y_true)
                loss.backward()

                #combined_loss = self.mse_weight * mse_loss + self.mae_weight * mae_loss
                #mse_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.optimizer.step()

                #total_mse_loss += mse_loss.item()
                #total_mae_loss += mae_loss.item()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            #avg_mse_loss = np.average(total_mse_loss)
            #avg_mae_loss = np.average(total_mae_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")#, Train MAE Loss: {avg_mae_loss:.4f}")

            # Validation Phase
            #val_mse_loss, val_mae_loss = self.validate(val_loader, criterion)
            val_loss = self.validate(val_loader, self.criterion)
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")#, Val MAE Loss: {val_mae_loss:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"Validation loss improved. Saving model at epoch {epoch+1}.")
                #torch.save(self.model.state_dict(), 'best_model.pth')
            #elif avg_mse_loss < best_val_loss:
            #    best_val_loss = avg_mse_loss
            #    patience_counter = 0
            #    print(f"Validation loss improved. Saving model at epoch {epoch+1}.")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience counter: {patience_counter}/{self.early_stopping_patience}")

            # Stop early if patience exceeds the limit
            if patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered. Training halted.")
                break

    def validate(self, val_loader, criterion):
        self.model.eval()
        #total_mse = 0
        #total_mae = 0
        total_loss = []
        with torch.no_grad():
            for batch in val_loader:
                x, y_true = [b.to(self.device) for b in batch]
                y_pred = self.model(x.unsqueeze(-1))#, None, y_true, None)  # Passing None for x_mark and y_mark

                #mse = self.mse_criterion(y_pred, y_true)
                #mae = self.mae_criterion(y_pred, y_true)
                loss = criterion(y_pred, y_true)
                
                #total_mse += mse.item()
                #total_mae += mae.item()
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        #avg_mse_loss = np.average(total_mse)
        #avg_mae_loss = np.average(total_mae)
        #return avg_mse_loss, avg_mae_loss
        self.model.train()
        return total_loss

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.model.eval()
        total_mse = 0
        total_mae = 0
        all_predictions = []
        all_true_values = []
        total_loss = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y_true = [b.to(self.device) for b in batch]
                y_pred = self.model(x.unsqueeze(-1))#, None, y_true, None)  # Passing None for x_mark and y_mark
                
                mse = self.mse_criterion(y_pred, y_true)
                mae = self.mae_criterion(y_pred, y_true)
                
                loss = self.criterion(y_pred, y_true)
                
                #total_mse += mse.item()
                #total_mae += mae.item()
                total_loss.append(loss.item())
                
                total_mse += mse.item()
                total_mae += mae.item()

                all_predictions.append(y_pred.cpu())
                all_true_values.append(y_true.cpu())
        
        avg_mse_loss = np.average(total_mse)
        avg_mae_loss = np.average(total_mae)
        
        avg_loss = np.average(total_loss)

        print(f"Test MSE: {avg_mse_loss:.4f}")
        print(f"Test MAE: {avg_mae_loss:.4f}")
        print(f"Test LOSS: {avg_loss:.4f}")

        # Concatenate all predictions and true values
        all_predictions = torch.cat(all_predictions, dim=0)
        all_true_values = torch.cat(all_true_values, dim=0)
        
        return all_predictions, all_true_values

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y_pred = self.model(x.unsqueeze(-1))#, None, None, None)  # Passing None for x_mark, y_true, and y_mark
        return y_pred.cpu()

    def set_dropout(self, dropout_rate):
        """
        Set dropout rate for all dropout layers in the model
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    def load_adj(dataset, adj_type: str = "doubletransition"):
            
        adj_mx = dataset
        if adj_type == "scalap":
            adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
        elif adj_type == "normlap":
            adj = [calculate_symmetric_normalized_laplacian(
                adj_mx).astype(np.float32).todense()]
        elif adj_type == "symnadj":
            adj = [calculate_symmetric_message_passing_adj(
                adj_mx).astype(np.float32).todense()]
        elif adj_type == "transition":
            adj = [calculate_transition_matrix(adj_mx).T]
        elif adj_type == "doubletransition":
            adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
        elif adj_type == "identity":
            adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
        elif adj_type == "original":
            adj = [adj_mx]
        else:
            error = 0
            assert error, "adj type not defined"
        return adj, adj_mx