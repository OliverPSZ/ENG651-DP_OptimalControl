% stochastic optimal control examples

% problem data
n = 5;
m = 2;
T = 30;

x0 = randn(n,1);

A = randn(n,n); A = A/max(abs(eig(A)));
B = randn(n,m);

%% optimal open-loop control sequence
% G = zeros(n*T,n);
% % H = zeros(n*T,m*T);
% H = eye(n*T);
% BB = kron(eye(T),B);
% 
% for i=1:T
%     G((i-1)*n+1:n*i,:) = A^i;
% end
% 
% for i=1:T
%     for j=1:T
%         if i > j
%             H((i-1)*n+1:n*i,(j-1)*n+1:n*j) = A^(i-j);
%         end
%     end
% end
% 
% HH = H*BB;
% 
% ustar = -(eye(T*m) + HH'*HH)\HH'*G*x0;


%% optimal closed-loop feedback policy
P = zeros(n,n,T);
K = zeros(m,n,T);
P(:,:,T+1) = eye(n);

for i=T:-1:1
    P(:,:,i) = eye(n) + A'*(P(:,:,i+1) - P(:,:,i+1)*B*inv(B'*P(:,:,i+1)*B + eye(m))*B'*P(:,:,i+1))*A;
    K(:,:,i) = -inv(B'*P(:,:,i+1)*B + eye(m))*B'*P(:,:,i+1)*A;
end

plot(0:29,reshape(K(1,1,:),T,1), 0:29, reshape(K(1,2,:),T,1), 0:29, reshape(K(1,3,:),T,1), 0:29, reshape(K(1,4,:),T,1), 0:29, reshape(K(1,5,:),T,1))
hold on;
plot(0:29,reshape(K(2,1,:),T,1), 0:29, reshape(K(2,2,:),T,1), 0:29, reshape(K(2,3,:),T,1), 0:29, reshape(K(2,4,:),T,1), 0:29, reshape(K(2,5,:),T,1))
xlabel('time');
ylabel('feedback gains');

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
Ns = 10000;
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

% cost of closed-loop policy
Jstarstoch_MC = mean(sum(Jstoch));
Jstarstoch = x0'*P(:,:,1)*x0;
for i=2:T+1
    Jstarstoch = Jstarstoch + trace(P(:,:,i));
end

[Jstarstoch, Jstarstoch_MC];
Jstar_cl = Jstarstoch

%% cost of open loop policy on stochastic system
% xol = zeros(n,T+1,Ns);
% uol = zeros(m,T,Ns);
% Jol = zeros(T+1,Ns);
% Jol(1,:) = x0'*x0;
% 
% for i=1:Ns
%     xol(:,1,i) = x0;
%     for j=2:T+1
%         uol(:,j-1,i) = ustar(j-1);
%         xol(:,j,i) = A*xol(:,j-1,i) + B*uol(:,j-1,i) + w(:,j-1,i);
%         Jol(j,i) = xol(:,j,i)'*xol(:,j,i) + uol(:,j-1,i)'*uol(:,j-1,i);
%     end
% end
% 
% Jstar_ol = mean(sum(Jol))
% 
% % plotting cost distributions
% subplot(2,1,1); hist(sum(Jol),50); 
% line([Jstar_ol Jstar_ol], ylim); 
% xlim([0,0.8*max(sum(Jol))]); 
% xlabel('open-loop cost distribution');
% 
% subplot(2,1,2); hist(sum(Jstoch),50); 
% 
% xlim([0,0.8*max(sum(Jol))]); 
% xlabel('closed-loop cost distribution');

