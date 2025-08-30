%% ------------------------------------------------------------------------
%
% Dynamic Mode Decomposition (DMD) for SOL-USD Price Prediction
%
% This script loads historical SOL-USD price data, applies DMD to model its
% underlying dynamics, and then uses the resulting model to forecast the
% next K price samples.
%
% -------------------------------------------------------------------------

clear; clc; close all;

%% --- 1. Load and Prepare Data ---

% Load your data from a CSV file.
% This example uses the provided 'sol_ada_data.csv'.
% To use your own data, change the filename and ensure the column name for
% the price data is correct in the line below.
data = readtable('sol_ada_data.csv');
sol_price = data.SOL_USD_Binance;
time_vector = data.datetime;

% Handle any missing data points by filling with the next available value.
sol_price = fillmissing(sol_price, 'next');

% --- User-Defined Parameters ---
K = 100;         % Number of future samples to predict.
r = 50 ;          % Rank for DMD truncation (a key tuning parameter).
dt = 1;          % Time step (assuming uniform sampling).


%% --- 2. Create Data Matrices with Hankel ---

% The DMD algorithm works with two matrices, X and X', that represent
% snapshots of the system's state at sequential time steps.

N = length(sol_price);

% Create the main data matrix 'X_AUG' using the Hankel function.
% This arranges the time series into a matrix of overlapping "windows".
X_AUG = hankel(sol_price(1:K), sol_price(K:N-1));

% Create the time-shifted matrix 'X_AUG_prime', which contains the
% subsequent state of each window in X_AUG.
X_AUG_prime = hankel(sol_price(2:K+1), sol_price(K+1:N));


%% --- 3. Perform Dynamic Mode Decomposition ---

% Use the provided DMD function to compute the modes and eigenvalues.
%
% Inputs:
%   - X_AUG:       The initial state data matrix.
%   - X_AUG_prime: The time-shifted data matrix.
%   - r:           The rank for SVD truncation.
%
% Outputs:
%   - Phi:      The DMD modes (spatial structures).
%   - omega:    The continuous-time eigenvalues.
%   - lambda:   The discrete-time eigenvalues.
%   - b:        The amplitudes of the DMD modes.
%   - Xdmd:     The reconstructed data from the DMD model.

[Phi, omega, lambda, b, Xdmd] = DMD2(X_AUG, X_AUG_prime, r, dt);


%% --- 4. Predict Future Prices ---

% Use the DMD model to forecast the price for the next K steps.
time_dynamics = zeros(r, length(time_vector));
for iter = 1:length(time_vector)
    time_dynamics(:,iter) = (b.*exp(omega*(iter)*dt));
end

% Reconstruct the full predicted time series from the time dynamics.
dmd_prediction_full = real(Phi * time_dynamics);

% Extract the first row, which corresponds to the predicted price.
dmd_sol_prediction = dmd_prediction_full(1, :);


%% --- 5. Visualize the Results ---

figure;
hold on;

% Plot the original SOL-USD price data.
plot(time_vector, sol_price, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Original SOL-USD');

% Plot the DMD prediction.
plot(time_vector, dmd_sol_prediction, 'r--', 'LineWidth', 2, 'DisplayName', 'DMD Prediction');

% Highlight the future prediction period.
xline(time_vector(end-K), 'k--', 'DisplayName', ['Prediction Start (', num2str(K), ' steps)']);

% Add labels and title for clarity.
title('SOL-USD Price Prediction using DMD');
xlabel('Time');
ylabel('Price (USD)');
legend('show');
grid on;
hold off;

% --- To help choose the rank 'r' ---
figure;
S = svd(X_AUG); % Get the singular values of the data matrix
semilogy(S,'o','LineWidth',2);
title('Singular Values of X_{AUG}');
xlabel('Rank');
ylabel('Singular Value Magnitude');
grid on;

%% --- Helper Function: DMD.m ---
% This is the DMD function from your provided files. It should be saved as
% 'DMD.m' in the same directory as this script.

function [Phi, omega, lambda, b, Xdmd] = DMD2(X,Xprime,r, dt)
    % --- Step 1: Singular Value Decomposition (SVD) ---
    [U, Sigma, V] = svd(X, 'econ');

    % --- Truncate the SVD matrices to rank 'r' ---
    Ur = U(:, 1:r);
    Sigmar = Sigma(1:r, 1:r);
    Vr = V(:, 1:r);

    % --- Step 2: Compute the Koopman operator approximation ---
    Atilde = Ur' * Xprime * Vr / Sigmar;

    % --- Step 3: Eigendecomposition of the Koopman operator ---
    [W, D] = eig(Atilde);
    lambda = diag(D); % Discrete-time eigenvalues

    % --- Step 4: Compute the DMD modes ---
    Phi = Xprime * (Vr / Sigmar) * W;

    % --- Compute continuous-time eigenvalues (growth/decay rates and frequencies) ---
    omega = log(lambda) / dt;

    % --- Compute mode amplitudes ---
    x1 = X(:, 1);
    b = Phi \ x1;

    % --- Reconstruct the data using the DMD model ---
    time_dynamics = zeros(r, size(X, 2));
    for iter = 1:size(X, 2)
        time_dynamics(:,iter) = (b.*exp(omega*(iter-1)*dt));
    end
    Xdmd = real(Phi * time_dynamics);
end