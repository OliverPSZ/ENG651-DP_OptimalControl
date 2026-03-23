% stochastic optimal control examples
close all;

% problem data
n = 12;
m = 4;
T = 30;

x0 = 40*randn(n,1);

A = randn(n,n); 
A = A/(0.+max(abs(eig(A))));
B = randn(n,m);

Q = 1*eye(n);
QT = 1*eye(n);
R = 1*eye(m);

W = 0.1*eye(n);

%% optimal closed-loop feedback policy
P = zeros(n,n,T+1);
r = zeros(1,T+1);
K = zeros(m,n,T);

tic;

P(:,:,T+1) = QT;

for i=T:-1:1
    P(:,:,i) = Q + A'*(P(:,:,i+1) - P(:,:,i+1)*B*inv(B'*P(:,:,i+1)*B + R)*B'*P(:,:,i+1))*A;
    r(i) = r(i+1) + trace(P(:,:,i+1)*W);
    K(:,:,i) = -inv(B'*P(:,:,i+1)*B + R)*B'*P(:,:,i+1)*A;
end

toc;

%% optimal open-loop control sequence
G = zeros(n*T,n);
% H = zeros(n*T,m*T);
H = eye(n*T);
BB = kron(eye(T),B*R^(-0.5));

for i=1:T
    G((i-1)*n+1:n*i,:) = Q^(0.5)*A^i;
end

for i=1:T
    for j=1:T
        if i > j
            H((i-1)*n+1:n*i,(j-1)*n+1:n*j) = Q^(0.5)*A^(i-j);
        end
    end
end

HH = H*BB;

ustar = -(eye(T*m) + HH'*HH)\HH'*G*x0;

%%
plot(0:T-1,reshape(K(1,1,:),T,1), 0:T-1, reshape(K(1,2,:),T,1), 0:T-1, reshape(K(1,3,:),T,1), 0:T-1, reshape(K(1,4,:),T,1), 0:T-1, reshape(K(1,5,:),T,1),'LineWidth',2)
xlabel('time');
ylabel('feedback gains');
legend('(K_t)_{11}', '(K_t)_{12}', '(K_t)_{13}', '(K_t)_{14}', '(K_t)_{15}')
set(gca, 'fontsize', 20);

% simulate without noise
x = zeros(n,T+1);
u = zeros(m,T);
J = zeros(T+1,1);
J(i) = x0'*x0;
x(:,1) = x0;

for i=2:T+1
    u(:,i-1) = K(:,:,i-1)*x(:,i-1);
    x(:,i) = (A + B*K(:,:,i-1))*x(:,i-1);
    J(i) = x(:,i)'*x(:,i) + u(:,i-1)'*u(:,i-1);
end

% cost of open loop policy on deterministic system
Jstar = sum(J);
x0'*P(:,:,1)*x0; % should be same as Jstar


% Monte Carlo simulation with Ns noise sequence samples
Ns = 1000;
xstoch = zeros(n,T+1,Ns);
ustoch = zeros(m,T,Ns);
Jstoch = zeros(T+1,Ns);
Jstoch(1,:) = x0'*x0;
w = randn(n,T,Ns);

for i=1:Ns
    xstoch(:,1,i) = x0;
    for j=2:T+1
        ustoch(:,j-1,i) = K(:,:,j-1)*xstoch(:,j-1,i);
        xstoch(:,j,i) = (A + B*K(:,:,j-1))*xstoch(:,j-1,i) + w(:,j-1,i);
        Jstoch(j,i) = xstoch(:,j,i)'*xstoch(:,j,i) + ustoch(:,j-1,i)'*ustoch(:,j-1,i);
    end
end

% plot sample trajectory
figure;
stairs(0:T,xstoch(:,:,1)','LineWidth',2)
set(gca, 'fontsize', 20);
xlabel('time');
ylabel('states');
title('sample state trajectory');

figure;
stairs(0:T-1,ustoch(:,:,1)','LineWidth',2)
set(gca, 'fontsize', 20);
xlabel('time');
ylabel('inputs');
title('sample input trajectory');


% cost of closed-loop policy
Jstarstoch_MC = mean(sum(Jstoch));
Jstarstoch = x0'*P(:,:,1)*x0;
for i=2:T+1
    Jstarstoch = Jstarstoch + trace(P(:,:,i));
end

[Jstarstoch, Jstarstoch_MC];
Jstar_cl = Jstarstoch_MC

%% cost of open loop policy on stochastic system
xol = zeros(n,T+1,Ns);
uol = zeros(m,T,Ns);
Jol = zeros(T+1,Ns);
Jol(1,:) = x0'*x0;

for i=1:Ns
    xol(:,1,i) = x0;
    for j=2:T+1
        uol(:,j-1,i) = ustar(j-1);
        xol(:,j,i) = A*xol(:,j-1,i) + B*uol(:,j-1,i) + w(:,j-1,i);
        Jol(j,i) = xol(:,j,i)'*xol(:,j,i) + uol(:,j-1,i)'*uol(:,j-1,i);
    end
end

Jstar_ol = mean(sum(Jol))

% plotting cost distributions
figure;
subplot(2,1,1); hist(sum(Jol),50); 
line([Jstar_ol Jstar_ol], ylim); 
xlim([0,max(sum(Jol))]); 
xlabel('open-loop cost distribution');
set(gca, 'fontsize', 20);

subplot(2,1,2); hist(sum(Jstoch),50); 
line([Jstar_cl Jstar_cl], ylim); 
xlim([0,max(sum(Jol))]); 
xlabel('closed-loop cost distribution');
set(gca, 'fontsize', 20);

