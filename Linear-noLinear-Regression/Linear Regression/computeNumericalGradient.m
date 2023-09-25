function numgrad = computeNumericalGradient(cost, theta)
% numgrad = computeNumericalGradient(cost, theta) computes the numerical
% gradient of the function cost at the point theta
%
% theta: a vector of parameters
% cost: a function that outputs a real-number. Calling y = cost(theta) will return the
% function value at theta.
%
% Escuela Politecnica Nacional
% Marco E. Benalcázar Palacios
% marco.benalcazar@epn.edu.ec

% Initialize numgrad with zeros
numgrad = zeros(numel(theta),1);
perturb = zeros(size(theta));
e = 1e-4;
for p = 1:numel(theta)
   disp(['Iteration: ' num2str(p) ' / ' num2str(numel(theta))]);
    % Set perturbation vector
    perturb(p) = e;
    loss1 = cost(theta - perturb);
    loss2 = cost(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end
end
