% distribution propagation and steady state examples

%% example 1: ergodic chain
% state transition matrix
P = [0.5 0.5 0 0; 
     0.3 0.4 0.3 0; 
     0 0.3 0.4 0.3; 
     0 0 0.5 0.5];

% random initial distribution
d0 = rand(1,4);  % random initial probabilities
d0 = d0/sum(d0) % make sure it sums to 1

T = 100;
d = zeros(T, 4);
d(1,:) = d0;

% distribution propagation
for t=1:T-1
    d(t+1,:) = d(t,:)*P;
end

dT = d(end,:)

%% example 2: absorbing chain
% state transition matrix
P = [0.3 0.3 0 0 0 0.4 0; 
     0.4 0.3 0.3 0 0 0 0; 
     0 0.4 0.3 0.3 0 0 0; 
     0 0 0.4 0.3 0.3 0 0; 
     0 0 0 0.4 0.3 0 0.3; 
     0 0 0 0 0 1 0; 
     0 0 0 0 0 0 1];

% random initial distribution
d0 = rand(1,7);  % random initial probabilities
d0 = d0/sum(d0) % make sure it sums to 1
%d0 = [1 0 0 0 0 0 0];
%d0 = [0 0 0 0 0 0 1];

T = 100;
d = zeros(T, 7);
d(1,:) = d0;

% distribution propagation
for t=1:T-1
    d(t+1,:) = d(t,:)*P;
end

dT = d(end,:)

%% example 3: periodic chain
P = [0 1; 1 0];

d0 = rand(1,2);
d0 = [1 0];

T = 100;
d = zeros(T, 2);
d(1,:) = d0;

% distribution propagation
for t=1:T-1
    d(t+1,:) = d(t,:)*P;
end

dT = d(end,:)
