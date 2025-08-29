% See Graph embedded dynamic mode  decomposition for stock price prediction
clear
clc

load("sol_ada.mat")
sol = sol_ada.SOL_USD_Binance;
ada = sol_ada.ADA_USD_Binance;
time= sol_ada.datetime;

sol = fillmissing(sol,"next");
ada = fillmissing(ada,"next");

K = 101

solana = sol(1:end-K);
N = numel(solana)

Ts = 1/100
Tf = 1
r = 30

SOL = hankel(solana(1:K), solana(K:N-1));
SOLp= hankel(SOL(2:K+1),SOL(K+1:N));

C = SOLp*pinv(SOL);
X = SOLp*pinv(SOL)*solana(end-(K-1):end);