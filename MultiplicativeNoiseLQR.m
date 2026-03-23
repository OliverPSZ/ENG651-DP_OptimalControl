% dynamic programming for linear systems with multiplicative noise

% problem data
n = 4;
m = 2;

% nominal system matrices (these are Abar and Bbar in the slides)
A = randn(n,n);
% scale down dynamics matrix to ensure mean-square stabilizability
A = 0.8*A/max(abs(eig(A))); 
B = randn(n,m);

% 
% friction = 10;  % nominal friction value
% A = [0 1; 
%      friction 1];
% 
% A1 = [0 0; 1 0];
% alpha1 = 1;  % variance of friction value

% multiplicative noise variances
SigmaA = randn(n^2, n^2); SigmaA = SigmaA*SigmaA'; 
SigmaA = 0.1*SigmaA/max(eig(SigmaA)); % scale down variances to ensure mean-square stabilizability
[Va,Ea] = eig(SigmaA);
a = diag(Ea); 
Aa = zeros(n,n,n^2);
for i=1:n^2
    Aa(:,:,i) = reshape(Va(:,i),n,n);
end

SigmaB = randn(n*m, n*m); SigmaB = SigmaB*SigmaB'; 
SigmaB = 0.1*SigmaB/max(eig(SigmaB)); % scale down variances to ensure mean-square stabilizability
[Vb,Eb] = eig(SigmaB);
b = diag(Eb); 
Bb = zeros(n,m,n*m);
for i=1:n*m
    Bb(:,:,i) = reshape(Vb(:,i),n,m);
end

Q = eye(n);
QT = eye(n);
R = eye(m);

T = 30;

%% finite horizon dynamic programming algorithm
P = zeros(n,n,T+1);
K = zeros(m,n,T);

P(:,:,T+1) = QT;

for i=T:-1:1
    P(:,:,i) = Q + MultSum(P(:,:,i+1), a, Aa) + A'*(P(:,:,i+1) - P(:,:,i+1)*B*inv(B'*P(:,:,i+1)*B + R)*B'*P(:,:,i+1))*A;
    K(:,:,i) = -inv(R + B'*P(:,:,i+1)*B + MultSum(P(:,:,i+1), b, Bb))*B'*P(:,:,i+1)*A;
end

%% infinite horizon
Pinf = zeros(n,n); % initialize
Pplus = randn(n,n); % data structure for optimal cost udpate

% value iteration recursion
t = 0;
while norm(Pinf - Pplus, 'fro') > 1e-6
    t = t + 1;
    Pinf = Pplus;
    Pplus = Q + A'*Pinf*A + MultSum(Pinf, a, Aa) - A'*Pinf*B*inv(R + B'*Pinf*B + MultSum(Pinf, b, Bb))*B'*Pinf*A;
end

Kinf = -inv(R + B'*Pinf*B + MultSum(Pinf, b, Bb))*B'*Pinf*A;

%% plot some stuff

plot(0:T-1,reshape(K(1,1,:),T,1), 0:T-1, reshape(K(1,2,:),T,1), 0:T-1, reshape(K(1,3,:),T,1), 0:T-1, reshape(K(1,4,:),T,1),'LineWidth',2)
xlabel('time');
ylabel('feedback gains');
legend('(K_t)_{11}', '(K_t)_{12}', '(K_t)_{13}', '(K_t)_{14}')
set(gca, 'fontsize', 20);


%% utility function for multiplicative noise variance terms
function Z = MultSum(X, z, Zz)
%%% 
    Z = zeros(size(Zz,2));
    for i=1:length(z)
        Z = Z + z(i)*Zz(:,:,i)'*X*Zz(:,:,i);
    end  
end