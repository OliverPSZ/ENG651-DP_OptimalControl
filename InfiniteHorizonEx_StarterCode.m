% Infinite horizon stopping problem

%% problem data
k = 20; % grid size
m = 2;
 
X = zeros(k);
X(5,5) = 1; X(17,10) = 1; X(10,15) = 1; % target states

G = zeros(k,k,m); % stage cost
G(:,:,1) = 1; % waiting cost
G(5,5,2) = -120; G(17,10,2) = -70; G(10,15,2) = -150; % stopping cost

%% DP algorithm
J = zeros(k,k); % optimal cost functions
pistar = zeros(k,k); % optimal policy

%%% YOUR INFINITE HORIZON DP CODE HERE %%%

% plot optimal policy and value function
colormap(flip(hot));
imagesc(pistar);

figure;
% colormap(flip(copper));
surf(J);
% shading interp