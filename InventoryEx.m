% Markov Chain inventory example

% problem data
C = 6;  % warehouse capacity
p = [0.1 0.2 0.7]; % demand pdf

% simulate
x0 = C;
T = 25;

% data structures to store state, input, and disturbance trajectories
x = zeros(T+1,1);
u = zeros(T,1);
d = zeros(T,1);
x(1) = x0;

for t=1:T
    u(t) = (C-x(t))*(x(t) <= 1);
    d(t) = sample(p) - 1;
    x(t+1) = min(max(x(t) - d(t) + u(t), 0), C);
end

plot(x,'o-', LineWidth=2);
hold on;
xlabel('time');
ylabel('inventory level');
set(gca,'FontSize',20);

figure;
plot(u,'x-');

figure; 
plot(d, 's-');
    

%% simulate with transition matrix (for C = 6)
C = 6;
p = flip(p);
P = [zeros(1,C-2) p;
     zeros(1,C-2) p;
     p 0 0 0 0;
     0 p 0 0 0;
     0 0 p 0 0;
     0 0 0 p 0;
     0 0 0 0 p];
 
for t=1:T
    x(t+1) = sample(P(x(t),:));
end
  
plot(x,'o-');
hold on;

%% distribution propagation
d0 = [zeros(1,C) 1];
d = zeros(T+1,C+1);
d(1,:) = d0;

for t=1:T
    d(t+1,:) = d(t,:)*P;
    bar(d(t,:)); ylim([0,1]);
end

dT = d(end,:)
bar(d(end,:)); ylim([0,1]);
xlabel('state at time 25');
ylabel('probability');

reorder_probability = d(end,:)*[1; 1; 0; 0; 0; 0; 0]



%% hitting time
p = [0.7 0.2 0.1];
p = flip(p); % change the probabilities to see how the hitting time distribution changes

Q = [1 0 0 0 0 0 0;
     0 1 0 0 0 0 0;
     p 0 0 0 0;
     0 p 0 0 0;
     0 0 p 0 0;
     0 0 0 p 0;
     0 0 0 0 p];

% compute re-order hitting time for t=1,...,50
t = 1:50;
Ptau = zeros(50,1);

for i=1:length(t)
    Z = Q^t(i) - Q^(t(i)-1);
    Ptau(i) = Z(7,1) + Z(7,2);
end

% expected hitting time
Etau = Ptau'*t';

plot(Ptau,'o-', 'LineWidth', 2);
xlabel('hitting time');
ylabel('probability');
title('hitting time distribution');
set(gca,'FontSize',20);
line([Etau Etau], ylim, 'LineWidth', 1, 'LineStyle', '--');