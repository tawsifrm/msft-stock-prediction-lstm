# Microsoft Stock Price Prediction

## Overview
This project utilizes machine learning techniques, specifically Long Short-Term Memory (LSTM) neural networks, to predict the closing prices of Microsoft stock. The historical stock data is preprocessed and used to train the model, which is then evaluated and tested for accuracy in predicting future stock prices.

## Tools Used
- Python 3.x
- TensorFlow and Keras for building and training the LSTM model
- Pandas for data manipulation
- Matplotlib and VBA for data visualization
- Scikit-learn for data preprocessing (StandardScaler)
- NumPy for numerical computations

## Data Preparation
The historical Microsoft stock data is loaded from a CSV file (`MicrosoftStock.csv`). Missing values are handled by dropping rows with missing data. The data is normalized using StandardScaler to bring all features to a standard scale.

## Model Building
- The training set is prepared by creating sequences of past 60 days' data as input features and the next day's closing price as the target variable.
- An LSTM neural network is built using TensorFlow and Keras. The model architecture includes two LSTM layers with 64 units each, followed by a Dense layer with ReLU activation and a Dropout layer to prevent overfitting. The output layer predicts the next day's closing price.

## Model Training
The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function. Root Mean Squared Error (RMSE) is used as a metric to evaluate the model's performance during training. The training data is split into training and validation sets (80-20 split), and the model is trained for 100 epochs with a batch size of 32.

## Model Testing and Evaluation
The trained model is tested on the remaining data not used for training. Sequences of past 60 days' data are used as input to predict the next day's closing price. The predictions are compared against the actual closing prices to evaluate the model's accuracy.

## Results
The project includes visualizations of both actual and predicted closing prices of Microsoft stock, allowing for a visual comparison of the model's performance.

![image](https://github.com/tawsifrm/msft-stock-prediction-lstm/assets/121325051/122688d4-4fbd-4e77-8844-ad2199d90938)

## Running the Code
1. Install the required libraries (`tensorflow`, `pandas`, `matplotlib`, `scikit-learn`) using `pip install`.
2. Ensure the CSV file containing Microsoft stock data is named `MicrosoftStock.csv` and located in the same directory as the code.
3. Run the code to train the model and generate predictions.

## VBA Script for Data Visualization
In addition to the Python-based visualization, a VBA script is provided for users who prefer to use Excel for data visualization. The VBA script reads the stock data from `MicrosoftStock.csv` and generates charts for actual and predicted stock prices.

### Using the VBA Script
1. Open Excel and press `Alt + F11` to open the VBA editor.
2. Insert a new module by clicking `Insert` > `Module`.
3. Copy and paste the VBA code from `StockVisualization.vba` into the module.
4. Run the script by pressing `F5`. Ensure that `MicrosoftStock.csv` is in the same directory as the Excel file.
