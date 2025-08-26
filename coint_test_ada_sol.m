clear
clc

% 1. Load data and run egcitest 
load("sol_ada.mat")
[Tstat,reg] = egcitest(sol_ada, 'rreg', 'ADF', 'creg','c','Lags', 1,'Test',"t2",'ResponseVariable','ADA_USD_Binance')

%[Tstat,reg] = egcitest(sol_ada, 'rreg', 'ADF', 'creg','c','Lags', 1,'Test',"t2",'ResponseVariable','SOL_USD_Binance')

% --- DATA PREPARATION
% 2. Get the residuals from the regression
residuals = reg.res;

% 3. Create a time vector that correctly corresponds to the residuals
%    It will be the LAST 'numel(residuals)' dates from the original timetable
time_vector = sol_ada.datetime((end - numel(residuals) + 1):end);

% 4. Create a correctly aligned timetable for the residuals
residuals_tt = timetable(time_vector, residuals);
residuals_tt.Properties.VariableNames = {'Residuals'}; % Give it a name

% 5. Now, create the lagged error correction term from this new timetable
ect = lag(residuals_tt, 1);
ect.Properties.VariableNames = {'ECT'}; % Rename for clarity

% 6. Take the first difference of the original data
diff_sol_ada = diff(sol_ada);

% 7. Synchronize the differenced data and the lagged error term
%    This will correctly align everything and drop any remaining NaNs
combined_data = synchronize(diff_sol_ada, ect, 'intersection');

% --- MODEL ESTIMATION (this part remains the same) ---

% 8. Define and estimate the VECM
model = varm(2, 2); % 2 variables, 2 lags
estModel = estimate(model, combined_data{:, {'SOL_USD_Binance', 'ADA_USD_Binance'}}, ...
    'X', combined_data.ECT);

% 9. Display the results
summarize(estModel);

% --- 4. Perform the Forecast (This creates YF_levels) ---
numPeriods = 10;
numSeries = 2; % SOL and ADA
% Get the pre-sample data needed to start the forecast
Y0 = sol_ada{end-1:end, {'SOL_USD_Binance', 'ADA_USD_Binance'}}; % Last 2 observations for a VAR(2) model

% Initialize matrices to store the forecast results
YF_diff = zeros(numPeriods, numSeries); % To store forecasted differences
YF_levels = zeros(numPeriods, numSeries); % To store forecasted levels
% Get the starting point for the levels forecast
last_known_level = sol_ada{end, {'SOL_USD_Binance', 'ADA_USD_Binance'}};
current_level = last_known_level;

% Get the very last known error correction term to start the loop
X0 = combined_data.ECT(end);

% --- Start the forecasting loop ---
for i = 1:numPeriods
    % a. Forecast one period ahead using the latest data
    [y_next_diff, ~] = forecast(estModel, 1, Y0, 'X', X0);
    
    % b. Store the forecasted difference
    YF_diff(i, :) = y_next_diff;
    
    % c. Calculate the next forecasted price LEVEL
    next_level = current_level + y_next_diff;
    YF_levels(i, :) = next_level;
    
    % d. Update the inputs for the NEXT loop iteration
    Y0 = [Y0(2:end, :); y_next_diff]; % Slide the observation window forward
    current_level = next_level; % Update the current level
    
    % e. Re-calculate the new error correction term using the cointegrating equation
    %    ECT = ADA - (c + beta * SOL)
    c = reg.coeff(1);
    beta = reg.coeff(2);
    X0 = current_level(2) - (c + beta * current_level(1)); 
end
% --- 5. Visualize the Results ---
disp('Script finished. Generating plots...');

% Create the multi-panel plot
figure;
t = tiledlayout(2, 2, 'TileSpacing', 'compact');
title(t, 'Cointegration and VECM Analysis: SOL vs ADA');

% Plot 1: Original Price Series
nexttile;
plot(sol_ada.datetime, sol_ada.SOL_USD_Binance, sol_ada.datetime, sol_ada.ADA_USD_Binance);
title('Original Price Series');
xlabel('Date');
ylabel('Price');
legend('SOL', 'ADA', 'Location', 'northwest');
grid on;

% Plot 2: Cointegrating Spread
nexttile;
plot(time_vector, reg.res);
yline(0, 'r--', 'LineWidth', 1.5);
title('Cointegrating Spread (Residuals)');
xlabel('Date');
ylabel('Spread');
grid on;

% Plot 3: VECM Forecast
nexttile;
num_hist_points = 100;
plot(sol_ada.datetime(end-num_hist_points:end), sol_ada{end-num_hist_points:end, :});
hold on;
last_time = sol_ada.datetime(end);
forecast_time = (last_time + days(1)):days(1):(last_time + days(numPeriods));
plot(forecast_time, YF_levels, 'LineWidth', 2);
title('VECM Forecast');
xlabel('Date');
ylabel('Price');
legend('SOL (Hist)', 'ADA (Hist)', 'SOL (Forecast)', 'ADA (Forecast)', 'Location', 'northwest');
grid on;
hold off;

% Plot 4: Impulse Response Function (in a new figure)
figure;
irf(estModel);
sgtitle('Impulse Response Function (IRF)');