%% ------------------------------------------------------------------------
%
% Constant Velocity Kalman Filter for 15-Minute Price Data
%
% This script is adapted for high-frequency (15-minute) data by correctly
% defining the time step 'dt' as a fraction of a day.
%
% -------------------------------------------------------------------------

clear; clc; close all;

%% --- 1. Load and Prepare Data ---

data = readtable('sol_ada_data.csv'); % Ensure this CSV has 15-minute data
sol_price = data.SOL_USD_Binance;
time_vector = data.datetime;

% Handle missing data
nan_indices = isnan(sol_price);
sol_price = sol_price(~nan_indices);
time_vector = time_vector(~nan_indices);

z = sol_price; % Measurements

%% --- 2. Split Data ---

split_ratio = 2/3;
split_idx = floor(length(z) * split_ratio);

train_data = z(1:split_idx);
validation_data = z(split_idx+1:end);
validation_length = length(validation_data);


%% --- 3. Define the Constant Velocity Kalman Filter Model ---

% --- CHANGE 1: Define dt as a fraction of a day ---
% The time step is now 15 minutes. There are 1440 minutes in a day.
dt = 15 / 1440;

% State transition matrix for constant velocity model
A = [1, dt; 0, 1];

% Measurement matrix (we only measure price)
H = [1, 0];

% --- Estimate Noise Covariances ---
% R: Measurement noise. Estimated from the variance of 15-minute price changes.
R = var(diff(train_data));

% Q: Process noise. Represents our uncertainty in the constant velocity model.
% The 'accel_variance' is the main tuning parameter for filter response.
accel_variance = 0.5; % TUNABLE: Increase for more responsiveness, decrease for more smoothness.
G = [dt^2/2; dt];     % Noise gain matrix
Q = G * G' * accel_variance;

% --- Initialize the Filter ---
% Initial state: [price; velocity]
initial_price = train_data(1);

% --- CHANGE 2: Scale initial velocity to be in units of $/day ---
% Calculate average change over 15 mins, then scale up to a full day.
intervals_in_day = 1440 / 15;
initial_velocity = mean(diff(train_data(1:10))) * intervals_in_day;
x_hat = [initial_price; initial_velocity];

% Initial error covariance
P = eye(2) * R;


%% --- 4. Run the Kalman Filter ---

% (This section requires no changes as 'A' and 'Q' are already updated)

% Prepare arrays to store results
prediction_1_step = zeros(validation_length, 1);
prediction_2_step = zeros(validation_length, 1); % Predicts 30 mins ahead

% Run over training data to let the filter converge
for k = 1:split_idx
    x_hat_minus = A * x_hat;
    P_minus = A * P * A' + Q;
    K_gain = P_minus * H' / (H * P_minus * H' + R);
    x_hat = x_hat_minus + K_gain * (train_data(k) - H * x_hat_minus);
    P = (1 - K_gain * H) * P_minus;
end

% Run over validation data, making predictions
for k = 1:validation_length
    % Predict 1 step (15 mins) into the future
    x_pred_1 = A * x_hat;
    prediction_1_step(k) = x_pred_1(1);

    % Predict 2 steps (30 mins) into the future
    x_pred_2 = A * x_pred_1;
    prediction_2_step(k) = x_pred_2(1);

    % Update the filter with the actual measurement for the current step
    P_minus = A * P * A' + Q;
    K_gain = P_minus * H' / (H * P_minus * H' + R);
    x_hat = x_pred_1 + K_gain * (validation_data(k) - H * x_pred_1);
    P = (1 - K_gain * H) * P_minus;
end


%% --- 5. Visualize the Results ---
% (No changes needed in this section)
figure('Position', [100, 100, 1200, 600]);
hold on;

train_time = time_vector(1:split_idx);
validation_time = time_vector(split_idx+1:end);

plot(train_time, train_data, 'b-', 'DisplayName', 'Training Data');
plot(validation_time, validation_data, 'g-', 'LineWidth', 2, 'DisplayName', 'Validation Data (Actual)');

plot(validation_time, prediction_1_step, 'r--', 'LineWidth', 1.5, 'DisplayName', '1-Step (15 min) Prediction');
plot(validation_time, prediction_2_step, 'm:', 'LineWidth', 1.5, 'DisplayName', '2-Step (30 min) Prediction');

xline(time_vector(split_idx), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Train/Validation Split');
title('Kalman Filter Multi-Step Prediction (15-Minute Data)');
xlabel('Date');
ylabel('Price (USD)');
legend('show', 'Location', 'northwest');
grid on;
hold off;