% stochastic optimal control for linear quadratic problems
close all;

% problem data
n = 5;
m = 2;
T = 30;

x0 = 40*randn(n,1);

A = randn(n,n); 
A = A/(max(abs(eig(A))));
B = randn(n,m);

Q = 1*eye(n);
QT = 5*eye(n);
R = 1*eye(m);

W = 0.1*eye(n);

%% optimal closed-loop feedback policy
P = zeros(n,n,T+1);
r = zeros(1,T+1);
K = zeros(m,n,T);

%%% YOUR DP CODE HERE %%%

%% plotting
% plot some interesting quantities!
% e.g., plot cost coefficient, gain coefficients, a sample trajectory, 
% compare with open-loop optimal control sequence, Monte Carlo optimal cost 
% estimate, etc.
