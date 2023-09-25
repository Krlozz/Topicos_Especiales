% This function checks the correctness of the computation of the gradient
% function of a linear regression model
%
% Escuela Politecnica Nacional
% Marco E. Benalcázar Palacios
% marco.benalcazar@epn.edu.ec
clc;
close all;
warning off all;

% Debugging the logistic regression classifier
n = 25; % Dimensión de los vectores
N = 100; % Número de vectores
dataX = randn(N, n);
dataY = rand(N, 1); 

% Randomly initialize beta
B = 0.005 * randn(n + 1,1);

% Regularization factor
lambda = 1e-3;

%  Implement cost function
[cost, grad] = linRegCostGrad(dataX, dataY, B, lambda);
numGrad = computeNumericalGradient( @(t) linRegCostGrad(dataX, dataY, t, lambda), B );

% Use this to visually compare the gradients side by side
disp([numGrad, grad]);

% Compare numerically computed gradients with those computed analytically
diff = norm(numGrad - grad)/norm(numGrad + grad);
disp(diff);
% The difference should be small.
% These values are usually less than 1e-7.

% When your gradients are correct, congratulations!
