% Basic Dynamic Programming Implementation for MDPs

%% problem data
n = 100;  % number of states
m = 20;   % number of inputs
T = 50;
 
P = rand(n,n,m);  % transition matrices

% normalize to get row stochastic matrices
for i=1:m
    P(:,:,i) = diag(1./sum(P(:,:,i),2))*P(:,:,i);
end

G = rand(n,m,T); % stage cost
GT = rand(n,1); % terminal cost

%% DP algorithm
J = zeros(n,T+1); % optimal cost functions
pistar = zeros(n,T); % optimal policy

% initialize
J(:,T+1) = GT;

tic; 


% recursion
for t = T:-1:1 % backward time recursion
    for i=1:n  % state loop 
        Ju = zeros(m,1);
        for j=1:m % input loop
            Ju(j) = G(i,j,t) + P(i,:,j)*J(:,t+1); 
        end
        [J(i,t), pistar(i,t)] = min(Ju);
    end
end

toc

figure; 
stairs(J(:,25), Linewidth=2);
xlabel('states');
ylabel('optimal value');

figure;
stairs(pistar(:,25), Linewidth=2);
xlabel('states');
ylabel('optimal action');
