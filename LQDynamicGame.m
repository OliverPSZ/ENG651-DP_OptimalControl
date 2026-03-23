% stochastic dynamic game via dynamic programming example
close all;

% problem data
n = 6;
m = 3;
p = 2;
T = 30;

x0 = 40*randn(n,1);

A = randn(n,n); 
A = A/(0.1+max(abs(eig(A))));
B = randn(n,m);
F = randn(n,p);

Q = 1*eye(n);
QT = 1*eye(n);
R = 1*eye(m);
gamma = 10;

W = 0.1*eye(n);

%% finite-horizon saddle point equilibrium cost and strategies
P = zeros(n,n,T+1);
r = zeros(1,T+1);
K = zeros(m,n,T);
L = zeros(p,n,T);

P(:,:,T+1) = QT;

for i=T:-1:1
    P(:,:,i) = Q + A'*P(:,:,i+1)*(eye(n) + (B*inv(R)*B' - gamma^2*(F*F'))*P(:,:,i+1))^(-1)*A;
    r(i) = r(i+1) + trace(P(:,:,i+1)*W);
    Qxu = A'*P(:,:,i)*B;
    Qxv = A'*P(:,:,i)*F;
    Quu = R + B'*P(:,:,i)*B;
    Quv = B'*P(:,:,i)*F;
    Qvv = F'*P(:,:,i)*F - gamma^2*eye(p);
    % check saddle point curvature condition
    if max(eig(Qvv)) > -1e-10
        disp('Warning: Saddle point curvature condition failed, equilibrium may not exist!');
    end
    K(:,:,i) = (Quu - Quv*Qvv^(-1)*Quv')^(-1)*(Quv*Qvv^(-1)*Qxv' - Qxu');
    L(:,:,i) = (Qvv - Quv'*Quu^(-1)*Quv)^(-1)*(Quv'*Quu^(-1)*Qxu' - Qxv');
end

%% infinite horizon dynamic game 
Pinf = zeros(n,n); % initialize
Pplus = randn(n,n); % data structure for optimal cost udpate

% gamma = 1;

% value iteration recursion to solve Isaacs Equation
t = 0; 
while norm(Pinf - Pplus, 'fro') > 1e-6
    t = t + 1;
    Pinf = Pplus;
    Pplus = Q + A'*Pinf*(eye(n) + (B*inv(R)*B' - gamma^(-2)*(F*F'))*Pinf)^(-1)*A;
    Qvv = F'*Pplus*F - gamma^2*eye(p);
    if max(eig(Qvv)) > -1e-10
        disp('Warning: Saddle point curvature condition failed, equilibrium may not exist!');
        break;
    end
end

Qxu = A'*Pinf*B;
Qxv = A'*Pinf*F;
Quu = R + B'*Pinf*B;
Quv = B'*Pinf*F;
Qvv = F'*Pinf*F - gamma^2*eye(p);

Kinf = (Quu - Quv*Qvv^(-1)*Quv')^(-1)*(Quv*Qvv^(-1)*Qxv' - Qxu');
Linf = (Qvv - Quv'*Quu^(-1)*Quv)^(-1)*(Quv'*Quu^(-1)*Qxu' - Qxv');

%% plot some stuff
% %%
% plot(0:T-1,reshape(K(1,1,:),T,1), 0:T-1, reshape(K(1,2,:),T,1), 0:T-1, reshape(K(1,3,:),T,1), 0:T-1, reshape(K(1,4,:),T,1), 0:T-1, reshape(K(1,5,:),T,1),'LineWidth',2)
% xlabel('time');
% ylabel('feedback gains');
% legend('(K_t)_{11}', '(K_t)_{12}', '(K_t)_{13}', '(K_t)_{14}', '(K_t)_{15}')
% set(gca, 'fontsize', 20);

