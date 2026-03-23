% Infinite horizon stopping problem

%% problem data
k = 20; % grid size
m = 2;
 
X = zeros(k);
X(5,5) = 1; X(17,10) = 1; X(10,15) = 1; % target states

G = zeros(k,k,m); % stage cost
G(:,:,1) = 1; % holding cost
G(5,5,2) = -12; G(17,10,2) = -70; G(10,15,2) = -150; % stopping cost

%% DP algorithm
J = zeros(k,k); % optimal cost functions
pistar = zeros(k,k); % optimal policy

Jplus = rand(k,k); % placeholder for optimal cost update
t = 0; % value iteration counter

% recursion
while norm(Jplus - J) > 1e-8 % value iteration
    t = t+1
    J = Jplus; 
    for i=1:k  % state loop 
        for j=1:k % state loop
            % interior states
            if (1 < i) && (i < k) && (1 < j) && (j < k)
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/4 1/4 1/4 1/4]*[J(i+1,j) J(i-1,j) J(i,j+1) J(i,j-1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
            
            % boundary states
            elseif (i == 1) && (1 < j) && (j < k) % top, can't move up
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/3 1/3 1/3]*[J(i+1,j) J(i,j+1) J(i,j-1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
                
            elseif (i == k) && (1 < j) && (j < k) % bottom, can't move down
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/3 1/3 1/3]*[J(i-1,j) J(i,j+1) J(i,j-1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
                
            elseif (j == 1) && (1 < i) && (i < k) % left, can't move left
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/3 1/3 1/3]*[J(i+1,j) J(i-1,j) J(i,j+1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
                
            elseif (j == k) && (1 < i) && (i < k) % right, can't move right
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/3 1/3 1/3]*[J(i+1,j) J(i-1,j) J(i,j-1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
                
            elseif (i == 1) && (j == 1) % top left corner
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/2 1/2]*[J(i+1,j) J(i,j+1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
                
            elseif (i == 1) && (j == k) % top right corner
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/2 1/2]*[J(i+1,j) J(i,j-1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
                
            elseif (i == k) && (j == 1) % bottom left corner
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/2 1/2]*[J(i-1,j) J(i,j+1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
                
            elseif (i == k) && (j == k) % bottom right corner
                Ju = zeros(m,1);
                Ju(1) = G(i,j,1) + [1/2 1/2]*[J(i-1,j) J(i,j-1)]';
                Ju(2) = G(i,j,2); 
                [Jplus(i,j), pistar(i,j)] = min(Ju);
            end
        end
    end
end

% plot optimal policy and value function
colormap(flip(hot));
imagesc(pistar);

figure;
% colormap(flip(copper));
surf(J);
% shading interp