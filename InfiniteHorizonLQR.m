% stochastic optimal control example
% infinite horizon LQR

% problem data
n = 12;
m = 4;

A = randn(n,n); A = A/max(abs(eig(A)));
B = randn(n,m);
% rank(ctrb(A,B)) % check for controllability 

Q = eye(n);
R = 1*eye(m);

W = 1*eye(n);

%% optimal closed-loop feedback policy
P = zeros(n,n); % initialize
Pplus = randn(n,n); % data structure for optimal cost udpate


% value iteration recursion
t = 0;
while norm(P - Pplus, 'fro') > 1e-6
    t = t + 1
    P = Pplus;
    Pplus = Q + A'*P*A - A'*P*B*inv(R + B'*P*B)*B'*P*A;
end

K = -inv(R + B'*P*B)*B'*P*A;

% check against built in ARE solver
[P1, K1, L1] = idare(A,B,Q,R);

norm(P - P1, 'fro')
norm(K + K1, 'fro')

%% Monte Carlo simulation with Ns noise sequence samples
Ns = 100;
T = 100;
x0 = 40*randn(n,1);
x = zeros(n,T+1,Ns);
u = zeros(m,T,Ns);
J = zeros(T+1,Ns);
J(1,:) = x0'*x0;
w = sqrt(1)*randn(n,T,Ns);

for i=1:Ns
    x(:,1,i) = x0;
    for j=2:T+1
        u(:,j-1,i) = K*x(:,j-1,i);
        x(:,j,i) = (A + B*K)*x(:,j-1,i) + w(:,j-1,i);
        J(j,i) = x(:,j,i)'*Q*x(:,j,i) + u(:,j-1,i)'*R*u(:,j-1,i);
    end
end

% plot sample trajectory
figure;
stairs(0:T,x(:,:,1)','LineWidth',2)
xlabel('time');
ylabel('states');
title('sample state trajectory');
set(gca, 'fontsize', 24);

figure;
stairs(0:T-1,u(:,:,1)','LineWidth',2)
xlabel('time');
ylabel('inputs');
title('sample input trajectory');
set(gca, 'fontsize', 24);

