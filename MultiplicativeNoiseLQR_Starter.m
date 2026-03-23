% dynamic programming for linear systems with multiplicative noise

% generate generic problem data
n = 4;
m = 2;

% nominal system matrices (these are Abar and Bbar in the slides)
A = randn(n,n); A = 0.8*A/max(abs(eig(A))); % scale down dynamics matrix to ensure mean-square stabilizability
B = randn(n,m);

% multiplicative noise variances
SigmaA = randn(n^2, n^2); SigmaA = SigmaA*SigmaA'; 
SigmaA = 0.1*SigmaA/max(eig(SigmaA)); % scale down variances to ensure mean-square stabilizability
[Va,Ea] = eig(SigmaA);
a = diag(Ea); % these are the alphai's
Aa = zeros(n,n,n^2); % this array contains all of the Ai's
for i=1:n^2
    Aa(:,:,i) = reshape(Va(:,i),n,n);
end

SigmaB = randn(n*m, n*m); SigmaB = SigmaB*SigmaB'; 
SigmaB = 0.1*SigmaB/max(eig(SigmaB)); % scale down variances to ensure mean-square stabilizability
[Vb,Eb] = eig(SigmaB);
b = diag(Eb); % these are the betai's
Bb = zeros(n,m,n*m); % this array contains all of the Bi's
for i=1:n*m
    Bb(:,:,i) = reshape(Vb(:,i),n,m);
end

% stage cost parameters
Q = eye(n);
QT = eye(n);
R = eye(m);

T = 30;

%% finite horizon dynamic programming algorithm
P = zeros(n,n,T+1);
K = zeros(m,n,T);

%%% YOUR CODE HERE %%%


%% infinite horizon
Pinf = zeros(n,n); % initialize
Pplus = randn(n,n); % data structure for optimal cost udpate

%%% YOUR CODE HERE %%%


%% plot/compare some stuff


