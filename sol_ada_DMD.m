clear
clc

load("sol_ada.mat")
sol = sol_ada.SOL_USD_Binance;
ada = sol_ada.ADA_USD_Binance;
time= sol_ada.datetime;

sol = fillmissing(sol,"next");
ada = fillmissing(ada,"next");
N = numel(sol)

Ts = 1/100
Tf = 1
r = 30
K = 2
t = time(1:end-K);

SOL = hankel(sol(1:K), sol(K:N-1));
SOLp= hankel(SOL(2:K+1),SOL(K+1:N));

C = SOLp*pinv(SOL)
X = SOLp*pinv(SOL)*sol;

[phi_s, omega_s, lambda_s, b_s, Xdmds,dynamics_s] = DMD(SOL,SOLp,r,Ts);
%[phi_a, omega_a, lambda_a, b_a, Xdmda,dynamics_a] = DMD(A,Ap,r,Ts);

%plot(dynamics_s)