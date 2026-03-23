% birth death chain example

% problem data
% transition matrix%
% P = [0.4 0.6 0 0 0 0;
%      0.3 0.1 0.6 0 0 0;
%      0 0.3 0.1 0.6 0 0;
%      0 0 0.3 0.1 0.6 0;
%      0 0 0 0.3 0.1 0.6;
%      0 0 0 0 0.3 0.7];
     
% metastable
P = [0.35 0.65 0 0 0 0 0 0;
     0.35 0 0.65 0 0 0 0 0;
     0 0.35 0 0.65 0 0 0 0;
     0 0 0.35 0 0.65 0 0 0;
     0 0 0 0.35 0 0.65 0 0;
     0 0 0 0 0.99 0 0.01 0;
     0 0 0 0 0 0.01 0 0.99;
     0 0 0 0 0 0 0.35 0.65];
 
% simulate
x0 = 1;
T = 1000;
x = zeros(T+1,1);
x(1) = x0;

for t=1:T
    % draw a sample from the discrete distribution corresponding to the
    % transition matrix row
    x(t+1) = sample(P(x(t),:));
end

plot(x,'o-');
hold on;
     