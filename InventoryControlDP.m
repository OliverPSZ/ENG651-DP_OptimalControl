% Basic Dynamic Programming Implementation for Inventory Control

%% problem data
C = 6;  % warehouse capacity
p = [0.1 0.2 0.7]; % demand pdf, Prob(wt = 2,1,0)

n = C + 1; % number of states
m = n;
T = 50; % time horizon

% cost parameters
s = 0.1; % unit stock storage cost
o = 1; % fixed ordering cost

g = [s*(0:C)' kron(s*(0:C)' + o, ones(1,C))]; % stage cost
gT = zeros(n,1);
% gT = s*(0:C)';


%% DP algorithm
J = zeros(n,T+1); % optimal cost functions
pistar = zeros(n,T); % optimal policy

% initialize
J(:,T+1) = gT;

% recursion
for t = T:-1:1 % backward time recursion
    for i=1:n  % state loop
        Ju = zeros(m,1) + inf;
        for j=max(4-i,1):min(n+1-i,n) % input loop, only update for allowable inputs
            pu = zeros(1,n);
            pu(j+i-3:j+i-1) = p;
            Ju(j) = g(i,j) + pu*J(:,t+1);
        end
        [J(i,t), pistar(i,t)] = min(Ju);
    end
end

% optimal cost
plot(0:C,J(:,1), 'LineWidth', 3);
set(gca,'fontsize',18);
xlabel('inventory level');

% optimal policy
figure;
bar(0:6, pistar(:,1) - 1);
ylabel('optimal order amount');
xlabel('inventory level');


