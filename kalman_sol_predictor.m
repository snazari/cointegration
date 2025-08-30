%% ------------------------------------------------------------------------
%
% Kalman Filter for SOL-USD Price Prediction
%
% This script uses a simple 1D Kalman filter to track and predict the price
% of SOL-USD. The model assumes a "random walk," where the price at the
% next time step is the current price plus some random noise. The filter
% provides an optimal estimate of the price by balancing this model
% with the actual measurements.
%
% -------------------------------------------------------------------------

clear; clc; close all;

%% --- 1. Load and Prepare Data ---

data = readtable('sol_ada_data.csv');
sol_price = data.SOL_USD_Binance;
time_vector = data.datetime;

% Handle any missing data by removing the rows.
nan_indices = isnan(sol_price);
sol_price = sol_price(~nan_indices);
time_vector = time_vector(~nan_indices);

% Get the full vector of price measurements
z = sol_price; % 'z' is the standard notation for measurements

%% --- 2. Split Data into Training and Validation Sets ---

split_ratio = 2/3;
split_idx = floor(length(z) * split_ratio);

train_data = z(1:split_idx);
validation_data = z(split_idx+1:end);
validation_length = length(validation_data);


%% --- 3. Define the Kalman Filter Model & Parameters ---

% For a simple random walk model (price_k = price_k-1 + noise), the
% state-space matrices are very simple.

A = 1; % State transition matrix: x_k = 1 * x_{k-1}
H = 1; % Measurement matrix: z_k = 1 * x_k

% --- Estimate Noise Covariances from Training Data ---
% Q: Process noise covariance. Represents the true volatility of the price.
% We estimate it as the variance of the daily price changes.
Q = var(diff(train_data));

% R: Measurement noise covariance. Represents the noise in the measurement
% itself. We assume the price data is quite accurate, so we set this to a
% smaller value relative to Q.
R = Q / 10; % This is a tunable parameter.

% --- Initialize the Filter ---
x_hat = train_data(1); % Initial state estimate (start with the first price)
P = Q;                 % Initial error covariance

prediction_horizon = 1; % Number of steps to predict ahead

% --- Prepare arrays to store results ---
x_hat_history = zeros(size(z));      % Stores the filtered estimate
prediction_history = nan(validation_length, 1); % Use NaN for multi-step predictions


%% --- 4. Run the Kalman Filter Recursively ---

% First, run the filter over the training data to let it "settle"
for k = 1:split_idx
    % Prediction Step
    x_hat_minus = A * x_hat;
    P_minus = A * P * A' + Q;

    % Update Step
    K_gain = P_minus * H' / (H * P_minus * H' + R);
    x_hat = x_hat_minus + K_gain * (train_data(k) - H * x_hat_minus);
    P = (1 - K_gain * H) * P_minus;

    x_hat_history(k) = x_hat;
end

% Now, run over the validation data, making multi-step-ahead predictions
for k = 1:prediction_horizon:validation_length
    % --- Update Step (Incorporate the actual measurement at step k) ---
    % First, predict to the current measurement point
    x_hat_minus = x_hat;
    P_minus = P;
    % Note: In the first loop (k=1), we use the state from the training phase.
    % For subsequent loops, x_hat was already propagated to the start of the horizon.

    % Update with the measurement at time k
    K_gain = P_minus * H' / (H * P_minus * H' + R);
    x_hat = x_hat_minus + K_gain * (validation_data(k) - H * x_hat_minus);
    P = (1 - K_gain * H) * P_minus;
    x_hat_history(split_idx + k) = x_hat;

    % --- Multi-step Prediction (Forecast for the next horizon) ---
    x_hat_pred = x_hat;
    for j = 1:prediction_horizon
        % Predict one step ahead
        x_hat_pred = A * x_hat_pred;

        % Store the prediction if it's within the validation data bounds
        if (k + j) <= validation_length
            prediction_history(k + j) = x_hat_pred;
        end
    end
    
    % The state for the next loop is the fully propagated state
    x_hat = x_hat_pred;
    % Propagate error covariance as well
    % A simplified propagation for the whole horizon:
    for j = 1:prediction_horizon, P = A * P * A' + Q; end
end


%% --- 5. Visualize the Results ---

figure('Position', [100, 100, 1200, 600]);
hold on;

% Define time vectors for plotting each segment
train_time = time_vector(1:split_idx);
validation_time = time_vector(split_idx+1:end);

% Plot the original data
plot(train_time, train_data, 'b-', 'DisplayName', 'Training Data');
plot(validation_time, validation_data, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Validation Data (Actual)');

% Plot the Kalman filter's one-step-ahead prediction
plot(validation_time, prediction_history, 'ro--', 'LineWidth', 2.0, 'DisplayName', 'Kalman Filter Prediction');

% Add styling and labels
xline(time_vector(split_idx), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Train/Validation Split');
title('Kalman Filter Price Prediction vs. Actuals');
xlabel('Date');
ylabel('Price (USD)');
legend('show', 'Location', 'northwest');
grid on;
hold off;

%% --- 6. Plot Prediction Error ---

figure('Position', [100, 100, 1200, 400]);
prediction_error = validation_data - prediction_history;
plot(validation_time, prediction_error, 'm-', 'DisplayName', 'Prediction Error');
title('Kalman Filter Prediction Error');
xlabel('Date');
ylabel('Error (USD)');
yline(0, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');
grid on;
legend('show', 'Location', 'northwest');

%% --- 7. Calculate and Display Error Metrics ---

% Calculate the variance of the prediction error, ignoring NaNs
variance_error = var(prediction_error, 'omitnan');

% Calculate the Mean Squared Error (MSE), ignoring NaNs
mse = mean(prediction_error.^2, 'omitnan');

% Display the results in the command window
fprintf('Prediction Error Variance: %.4f\n', variance_error);
fprintf('Mean Squared Error (MSE):  %.4f\n', mse);