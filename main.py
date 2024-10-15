import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from ModelTrainer import ModelTrainer
from SegRNN import SegRNN, Config  # You need to import your model and configs

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return x, y

# Load and preprocess data
date_rng = pd.date_range(start='2022-01-01', end='2022-12-31', freq='H')
df = pd.DataFrame(date_rng, columns=['timestamp'])

# Feature columns
df['electricity_consumption'] = np.random.normal(1000, 100, size=(len(date_rng)))
df['temperature'] = np.random.normal(20, 5, size=(len(date_rng)))
df['humidity'] = np.random.normal(60, 10, size=(len(date_rng)))
df['wind_speed'] = np.random.normal(10, 3, size=(len(date_rng)))
df['solar_radiation'] = np.random.normal(500, 100, size=(len(date_rng)))

# Select features (excluding timestamp)
features = df.columns.to_list()[1:]
print(features)
#features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']  # Replace with your actual feature names

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[features])

# Split the data into train and test sets
train_data = normalized_data[:int(0.8*len(normalized_data))]
test_data = normalized_data[int(0.8*len(normalized_data)):]

# Set up the configurations
configs = Config()
configs.enc_in = len(features)  # number of input variables
configs.seq_len = 96  # input sequence length (e.g., 4 days of hourly data)
configs.pred_len = 24  # prediction sequence length (e.g., next day prediction)
configs.patch_len = 24  # patch length (e.g., daily patches)
configs.d_model = 512  # model dimension
configs.dropout = 0.1  # dropout rate

# Create datasets
train_dataset = TimeSeriesDataset(train_data, configs.seq_len, configs.pred_len)
test_dataset = TimeSeriesDataset(test_data, configs.seq_len, configs.pred_len)

# Initialize the model
model = SegRNN(configs)

# Create trainer instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = ModelTrainer(model, learning_rate=0.001, batch_size=32, device=device)

print(f"Using device: {trainer.device}")

# Train the model
trainer.train(train_dataset, num_epochs=10)

# Test the model
all_predictions, all_true_values = trainer.test(test_dataset)

# Make predictions
def prepare_prediction_data(data, seq_len):
    return torch.FloatTensor(data[-seq_len:]).unsqueeze(0)  # Add batch dimension

# Predict the next pred_len time steps
last_sequence = normalized_data[-configs.seq_len:]
x_pred = prepare_prediction_data(last_sequence, configs.seq_len)
predictions = trainer.predict(x_pred)

# Denormalize predictions
denormalized_predictions = scaler.inverse_transform(predictions.squeeze().numpy())

# Print predictions
print("\nPredictions for the next {} time steps:".format(configs.pred_len))
for i, pred in enumerate(denormalized_predictions):
    print(f"Step {i+1}: {pred}")

# Optionally, you can save the model
#torch.save(model.state_dict(), 'trained_model.pth')

# To load the model later:
# loaded_model = Model(configs)
# loaded_model.load_state_dict(torch.load('trained_model.pth'))