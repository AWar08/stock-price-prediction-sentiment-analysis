"""
LSTM Stock Volume Prediction
----------------------------
This script trains an LSTM model to predict Apple stock trading volumes
using historical data. It also evaluates the model against a naive baseline
and visualizes actual vs predicted values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import custom LSTM implementation
from lstm import LstmParam, LstmNetwork


# Loss Layer for LSTM Training
class ToyLossLayer:
    @classmethod
    def loss(cls, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(cls, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


# Create input sequences for time series
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return sequences, labels


def train_and_predict(data_path, seq_length=600, epochs=100, pred_days=1000):
    # Load and preprocess data
    data = pd.read_csv(data_path)
    volume_traded = data['Volume Traded'].values
    normalized_volume = volume_traded / np.max(volume_traded)

    # Create sequences
    sequences, labels = create_sequences(normalized_volume, seq_length)

    # LSTM setup
    mem_cell_ct = 100
    x_dim = seq_length
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    # Training loop
    for cur_iter in range(epochs):
        for ind in range(len(labels)):
            lstm_net.x_list_add(sequences[ind])
            loss = lstm_net.y_list_is([labels[ind]], ToyLossLayer)
            lstm_param.apply_diff(lr=0.1)
            lstm_net.x_list_clear()
        print(f"Iteration {cur_iter}, Loss: {loss:.3e}")

    # Prediction
    predicted_volume = []
    current_sequence = normalized_volume[-seq_length:]
    for _ in range(pred_days):
        lstm_net.x_list_add(current_sequence)
        predicted = lstm_net.lstm_node_list[-1].state.h[0]
        predicted_volume.append(predicted * np.max(volume_traded))
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted
        lstm_net.x_list_clear()

    return predicted_volume, volume_traded


def calculate_accuracy_score(model_error, baseline_error):
    improvement = max(0, baseline_error - model_error) / baseline_error
    return improvement * 100


if __name__ == "__main__":
    data_path = "../data/Apple_Trading_Volume_1000_Days_All_Days.csv"

    predicted_volume, actual_volume = train_and_predict(data_path)

    # Evaluation
    actual_tail = actual_volume[-len(predicted_volume):]

    lstm_mse = mean_squared_error(actual_tail, predicted_volume)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mae = mean_absolute_error(actual_tail, predicted_volume)

    # Naive baseline (previous value as prediction)
    naive_predictions = actual_volume[-len(predicted_volume) - 1:-1]
    adjusted_actual_volume = actual_volume[-len(naive_predictions):]

    naive_mse = mean_squared_error(adjusted_actual_volume, naive_predictions)
    naive_rmse = np.sqrt(naive_mse)
    naive_mae = mean_absolute_error(adjusted_actual_volume, naive_predictions)

    mse_score = calculate_accuracy_score(lstm_mse, naive_mse)
    rmse_score = calculate_accuracy_score(lstm_rmse, naive_rmse)
    mae_score = calculate_accuracy_score(lstm_mae, naive_mae)
    average_score = (mse_score + rmse_score + mae_score) / 3

    print("Model Accuracy based on MSE:", mse_score)
    print("Model Accuracy based on RMSE:", rmse_score)
    print("Model Accuracy based on MAE:", mae_score)
    print("Overall Model Accuracy:", average_score)

    # Visualization
    plt.figure(figsize=(15, 6))
    plt.plot(actual_volume, label='Actual Trading Volume')
    plt.plot(range(len(actual_volume), len(actual_volume) + len(predicted_volume)),
             predicted_volume, label='Predicted Trading Volume', color='red')
    plt.title('Trading Volume: Actual vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()

