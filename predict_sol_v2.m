%% ------------------------------------------------------------------------
%
% DMD for SOL-USD Prediction with Mode Selection and Train/Validation
%
% This script applies best practices for DMD forecasting on financial data:
% 1. Pre-processes the data using log-differencing to achieve stationarity.
% 2. Splits data into a training set and a validation set.
% 3. Performs DMD on the training data.
% 4. **Selects only stable modes (|lambda| <= 1) to build a robust model.**
% 5. Performs a recursive forecast over the validation period.
% 6. Inverse transforms the forecast to get the final price prediction.
%
% -------------------------------------------------------------------------

clear; clc; close all;

%% --- 1. Load and Prepare Data ---

data = readtable('sol_ada_data.csv');
sol_price = data.SOL_USD_Binance;
time_vector = data.datetime;

% Handle missing data
sol_price = rmmissing(sol_price);
time_vector = time_vector(~isnan(data.SOL_USD_Binance));


%% --- 2. Pre-process the Data ---

sol_price_log = log(sol_price);
processed_data = diff(sol_price_log);


%% --- 3. Split Data into Training and Validation Sets ---

split_ratio = 2/3;
split_idx = floor(length(processed_data) * split_ratio);

train_data = processed_data(1:split_idx);
validation_data = processed_data(split_idx+1:end);
validation_length = length(validation_data);


%% --- 4. Build DMD Model on the Training Set ---

% --- Hyperparameters ---
K = 100; % Window size (embedding dimension)
r = 40;  % Rank for truncation. This is a key parameter to tune.
dt = 1/100;  % Time step

N_train = length(train_data);

X_AUG = hankel(train_data(1:K), train_data(K:N_train-1));
X_AUG_prime = hankel(train_data(2:K+1), train_data(K+1:N_train));

% Perform the standard DMD computation
[Phi, omega, lambda, b, ~] = DMD(X_AUG, X_AUG_prime, r, dt);


%% --- 5. Mode Selection for a Stable Forecast ---

% This is the CRITICAL new step.
% Unstable modes (eigenvalues > 1) will cause the forecast to explode.
% We select only the stable modes for our predictive model.

stable_indices = find(abs(lambda) <= 1.0);

% Keep only the stable modes, eigenvalues, and their initial amplitudes
Phi_stable = Phi(:, stable_indices);
lambda_stable = lambda(stable_indices);


%% --- 6. Forecast Over the Validation Period (Recursive Method) ---

predicted_log_returns = zeros(1, validation_length);
current_state = train_data(end-K+1:end);

for i = 1:validation_length
    % Project the current state onto the STABLE modes to find amplitudes
    b_current = pinv(Phi_stable) * current_state;

    % Evolve the amplitudes forward one step using the STABLE eigenvalues
    b_next = b_current .* lambda_stable;

    % Reconstruct the next state from the stable modes
    next_state = real(Phi_stable * b_next);

    % The prediction is the first element of the new state
    predicted_log_returns(i) = next_state(1);

    % Update the state for the next iteration (roll the window)
    current_state = [current_state(2:end); predicted_log_returns(i)];
end


%% --- 7. Inverse Transform and Visualize ---

last_train_log_price = sol_price_log(split_idx + 1);
predicted_log_prices = cumsum(predicted_log_returns) + last_train_log_price;
predicted_prices = exp(predicted_log_prices);

% --- Plotting ---
figure('Position', [100, 100, 1200, 600]);
hold on;

train_time = time_vector(1:split_idx+1);
validation_time = time_vector(split_idx+2:end);

plot(train_time, sol_price(1:split_idx+1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Training Data');
plot(validation_time, sol_price(split_idx+2:end), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Validation Data (Actual)');
plot(validation_time, predicted_prices, 'r--', 'LineWidth', 2.0, 'DisplayName', 'Stable DMD Prediction');

xline(train_time(end), 'k--', 'LineWidth', 1.5, 'DisplayName', 'Train/Validation Split');
title('Stable DMD Price Prediction vs. Actuals');
xlabel('Time');
ylabel('Price (USD)');
legend('show', 'Location', 'northwest');
grid on;
hold off;
ylim([-300 300])


%% --- Helper Function: DMD.m ---

function [Phi, omega, lambda, b, Xdmd] = DMD(X, Xprime, r, dt)
    [U, Sigma, V] = svd(X, 'econ');
    Ur = U(:, 1:r);
    Sigmar = Sigma(1:r, 1:r);
    Vr = V(:, 1:r);
    Atilde = Ur' * Xprime * Vr / Sigmar;
    [W, D] = eig(Atilde);
    lambda = diag(D);
    Phi = Xprime * (Vr / Sigmar) * W;
    omega = log(lambda) / dt;
    x1 = X(:, 1);
    b = pinv(Phi) * x1; % Using pseudoinverse for robustness
    time_dynamics = zeros(r, size(X, 2));
    for iter = 1:size(X, 2)
        time_dynamics(:,iter) = (b.*exp(omega*(iter-1)*dt));
    end
    Xdmd = real(Phi * time_dynamics);
end