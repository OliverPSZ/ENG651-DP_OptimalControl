% absorbing Markov Chain

% problem data
p = [0.4 0.3 0.3];
P = [p 0 0 0 0;
     0 p 0 0 0;
     0 0 p 0 0;
     0 0 0 p 0;
     0 0 0 0 p;
     0 0 0 0 0 1 0;
     0 0 0 0 0 0 1];
P11 = P(1:5,1:5);
P12 = P(1:5,6:7);
P22 = P(6:7,6:7);
 
% limit matrix
L = (eye(5) - P11)\P12
