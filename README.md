# Stock Price Forecasting with LSTM

Forecasting Tesla (TSLA) daily closing prices with a Long Short-Term Memory (LSTM) neural network built in TensorFlow/Keras.

The project walks through the full workflow: loading and inspecting the data, scaling, building sliding-window sequences, training an LSTM with early stopping, evaluating with RMSE, and plotting predictions against real values.

## Dataset

- **Source:** [Tesla Stock Data from 2010 to 2020 (Kaggle)](https://www.kaggle.com/timoboz/tesla-stock-data-from-2010-to-2020)
- **File:** `TSLA.csv`
- **Columns:**

| Column | Description |
|---|---|
| `Date` | Trading day |
| `Open` | Opening price |
| `High` | Highest price of the day |
| `Low` | Lowest price of the day |
| `Close` | Closing price |
| `Adj Close` | Adjusted closing price |
| `Volume` | Trading volume |

Only the `Close` column is used as the modeling target (univariate forecasting).

## Approach

1. **Data inspection**: shape, dtypes, head/tail, missing values and quantiles via a `check_df` helper.
2. **Preprocessing**: parse `Date`, set it as the index, keep `Close`, cast to `float32`.
3. **Chronological split**: first 80% of the series for training, last 20% for testing (no shuffling).
4. **Scaling**: `MinMaxScaler` to the range (0, 1).
5. **Windowing**: a lookback window of 20 days is used to predict the next day's close.
6. **Model**: 

   | Layer | Details |
   |---|---|
   | LSTM | 50 units, ReLU activation |
   | Dropout | 0.2 |
   | Dense | 1 output |

7. **Training**: Adam optimizer, MSE loss, batch size 20, up to 100 epochs, with
   - `EarlyStopping` (monitor `val_loss`, patience 3)
   - `ModelCheckpoint` (best model saved to `mymodel.h5`)
8. **Evaluation**: predictions are inverse-transformed back to price scale and evaluated with RMSE on train and test sets.
9. **Visualization**: training/validation loss curves, and real vs. predicted prices over time.

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Usage

The script was exported from a Google Colab notebook and uses `google.colab.files.upload()` to load the data.

**On Google Colab**

1. Open the notebook / run the script.
2. When prompted, upload `TSLA.csv`.
3. Run all cells.

**Locally**

1. Download `TSLA.csv` from Kaggle and place it next to the script.
2. Remove (or comment out) these lines:
   ```python
   from google.colab import files
   upload = files.upload()
   ```
3. Run:
   ```bash
   python stockpriceforecastinglstm.py
   ```

## Outputs

- Training vs. validation loss plot
- Train and test RMSE (in price units)
- Plot of real prices with train predictions (blue) and test predictions (red)
- Best model checkpoint: `mymodel.h5`

## Results

| Metric | Value |
|---|---|
| Train RMSE | _fill in after running_ |
| Test RMSE | _fill in after running_ |

## Limitations and Possible Improvements

- **Univariate input:** only `Close` is used. Adding `Open`, `High`, `Low`, `Volume` or technical indicators (moving averages, RSI) could help.
- **Separate scalers for train and test:** the scaler is fit independently on each split. A more standard approach is to fit the scaler on the training data only and reuse it for the test data, so both are on the same scale and no information from the test set is used.
- **Reshaping:** the input is shaped as `(samples, 1, 20)`, i.e. one timestep with 20 features. Using `(samples, 20, 1)` would let the LSTM treat the window as a true sequence.
- **Activation:** the LSTM uses `relu`; the default `tanh` is usually more stable for LSTMs.
- **Validation:** the test set doubles as validation data for early stopping, so the reported test score is slightly optimistic. A separate validation split or walk-forward validation would be more rigorous.
- **Baseline:** comparing against a naive "tomorrow = today" baseline gives useful context for the RMSE.
- **Single-step forecasts only:** the model predicts one day ahead from real past values, not a multi-day horizon.
- **Disclaimer:** this is an educational project, not financial advice.

## Project Structure

```
.
├── stockpriceforecastinglstm.py   # Main script (exported from Colab)
├── TSLA.csv                       # Dataset (download from Kaggle)
├── mymodel.h5                     # Saved best model (generated on training)
└── README.md
```
