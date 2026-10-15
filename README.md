# Stock Price Forecasting with LSTM

Forecasting Tesla (TSLA) daily closing prices with a Long Short-Term Memory (LSTM) neural network built in TensorFlow/Keras.

The project covers the full workflow: data inspection, leakage-free scaling, sliding-window sequences, training with early stopping, evaluation against a naive baseline, and visualization of predictions.

## Dataset

- **Source:** [Tesla Stock Data from 2010 to 2020 (Kaggle)](https://www.kaggle.com/timoboz/tesla-stock-data-from-2010-to-2020)
- **File:** `TSLA.csv`

| Column | Description |
|---|---|
| `Date` | Trading day |
| `Open` | Opening price |
| `High` | Highest price of the day |
| `Low` | Lowest price of the day |
| `Close` | Closing price |
| `Adj Close` | Adjusted closing price |
| `Volume` | Trading volume |

Only the `Close` column is used (univariate forecasting).

## Approach

1. **Data inspection**: shape, dtypes, head/tail, missing values, quantiles (`check_df`).
2. **Chronological split** (no shuffling): 68% train / 12% validation / 20% test.
3. **Scaling**: `MinMaxScaler` fit on the training part only, then applied to all splits.
4. **Windowing**: the previous 20 days form one sequence of shape `(20, 1)` and predict the next day's close. Windows are built over the full series, so test predictions use the last training days as context and no test points are lost.
5. **Model**:

   | Layer | Details |
   |---|---|
   | LSTM | 50 units (default `tanh`) |
   | Dropout | 0.2 |
   | Dense | 1 output |

6. **Training**: Adam, MSE loss, batch size 32, up to 100 epochs, with
   - `EarlyStopping` (monitor `val_loss`, patience 10, restores best weights)
   - `ModelCheckpoint` (best model saved to `mymodel.keras`)
   - fixed random seed for reproducibility
7. **Evaluation**: predictions are inverse-transformed to price scale. RMSE, MAE and MAPE are reported for train, validation and test. The test set is used only for the final evaluation.
8. **Baseline**: a naive forecast ("tomorrow = today") is evaluated on the same test targets, and the script reports whether the LSTM beats it.
9. **Visualization**: loss curves and real vs. predicted prices (train / validation / test).

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib, scikit-learn, tensorflow

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Usage

**Locally**

1. Download `TSLA.csv` from Kaggle and place it next to the script.
2. Run:
   ```bash
   python stockpriceforecastinglstm.py
   ```

**Google Colab**

Run the script in a cell. If `TSLA.csv` is not found, an upload dialog opens automatically.

## Outputs

- Training vs. validation loss plot
- Metrics table (RMSE / MAE / MAPE) for LSTM on train, validation, test, plus the naive baseline on test
- Plot of real prices with train, validation and test predictions
- Best model checkpoint: `mymodel.keras`

## Results

Run on the full dataset (2,416 trading days, 2010-06-29 to 2020-02-03). Split sizes (windows): 1,623 train / 290 validation / 483 test. Early stopping triggered at epoch 22 and restored the weights from epoch 12.

| Model | Split | RMSE | MAE | MAPE |
|---|---|---|---|---|
| LSTM | Train | 7.684 | 5.868 | 7.57% |
| LSTM | Validation | 15.839 | 12.898 | 3.95% |
| LSTM | Test | 26.090 | 16.583 | 4.98% |
| Naive baseline ("tomorrow = today") | Test | **12.386** | **7.637** | **2.46%** |

**The LSTM does not beat the naive baseline on the test set** (test RMSE is roughly twice as high).

### Interpretation

- **Train / validation / test are on very different price levels.** The series rises from about 16 to 780 USD. The scaler is fit on the training part only, so test prices fall well outside the (0, 1) range seen during training, and the LSTM struggles to extrapolate to levels it has never seen. (With a separate scaler per split, as in the original notebook, this problem was hidden and the test error looked better than it really was.)
- **Validation loss is noisy** (between about 0.003 and 0.020) while training loss decreases steadily, which suggests the model learns the training price regime but generalizes poorly to later periods.
- **Train MAPE is higher than validation MAPE** because early prices are low (around 20 USD), so the same dollar error is a larger percentage. RMSE and MAE are the more comparable numbers across splits.
- **Price-level forecasting is a hard target.** On trending series, "yesterday's price" is already a strong predictor, and a model needs a real signal to improve on it.

These explanations are likely reasons based on the training behavior and data, not proven causes.

## Limitations and Possible Improvements

- **Univariate input:** adding `Volume`, `High`, `Low` or technical indicators (moving averages, RSI) could help.
- **Price-level target (next step):** forecasting log returns (`np.log(close).diff()`) instead of prices would remove the scale problem described above, since returns are roughly stationary. Predictions can be converted back to prices and compared with the naive baseline, and directional accuracy can be added as a metric.
- **Single-step forecast:** the model predicts one day ahead from real past values, not a multi-day horizon.
- **Single split:** walk-forward (rolling) validation would give a more robust estimate than one train/val/test split.
- **Hyperparameters:** lookback, units, dropout and learning rate are not tuned.
- **Disclaimer:** this is an educational project, not financial advice.

## Project Structure

```
.
├── stockpriceforecastinglstm.py   # Main script
├── TSLA.csv                       # Dataset (download from Kaggle)
├── mymodel.keras                  # Saved best model (generated on training)
└── README.md
```
