%function [Phi, Lambda, b] = DMD(X,Xprime,r)
% [U,Sigma,V] = svd(X,'econ'); % Step 1
% Ur = U(:,1:r);
% Sigmar = Sigma(1:r,1:r);
% Vr = V(:,1:r);
% Atilde = Ur'*Xprime*Vr/Sigmar; % Step 2
% [W,Lambda] = eig(Atilde); % Step 3
% Phi = Xprime*(Vr/Sigmar)*W; % Step 4
% alpha1 = Sigmar*Vr(1,:)';
% b = (W*Lambda)\alpha1;
function [Phi ,omega ,lambda ,b,Xdmd,time_dynamics] = DMD(X1,X2,r,dt)
disp('Beginning DMD...')
[U, S, V] = svd(X1, 'econ');
r = min(r, size(U,2));
U_r = U(:, 1:r); % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);
Atilde = U_r' * X2 * V_r / S_r ; % low-rank dynamics
[W_r , D] = eig(Atilde);
Phi = X2 * V_r / S_r * W_r ; % DMD modes
lambda = diag(D); % discrete -time eigenvalues
omega = log(lambda)/dt; % continuous-time eigenvalues
x1 = X1(:, 1);
b = Phi\x1;
mm1 = size (X1 , 2); % mm1 = m - 1
time_dynamics = zeros(r, mm1);
t = (0: mm1 -1) *dt; % time vector
for iter = 1:mm1;
    time_dynamics (:,iter ) = (b.*exp(omega*t(iter )));
end;
Xdmd = Phi * time_dynamics;
disp('Done.')

