function [Z, A] = ffnnRegTest(dataX, W, transferFunctions, options)
%% This program computes the response of an artificial neural network for
% regression
%
% Inputs:
% dataX                   [N n] matrix, where each row contains an observation
%                         X = (x_1, x_2,...,x_n)
%
% W                       Cell containing the weights of each layer of the
%                         network, where W{1} contains the weights of
%                         the connections between the input and the first
%                         hidden layer, W{2} contains the weights of
%                         the connections between the first and the second
%                         hidden layer, and so on
%
% transferFunctions       Cell containg the name of the transfer functions
%                         of each layer of the neural network. Options of transfer
%                         functions are:
%                         - none: input layer has no transfer functions
%                         - tanh: hyperbolic tangent
%                         - elu: exponential linear unit
%                         - softplus: log(exp(x) + 1)
%                         - relu: rectified linear unit
%                         - logsig: logistic function
%                         - purelin: f(x) = x
%
% options                structure containing additional settings for the
%                        neural network (e.g., rectified linear unit
%                        threshold)
%
% Outputs:
% Z                     Cell containing the multiplicaction between the inputs
%                       and weights of each layer (this is used for the
%                       back-propagation algorithm)
%
% A                     Cell containing the response of the activation functions
%                       of each layer
%
% Escuela Politecnica Nacional
% Marco E. Benalc�zar Palacios
% marco.benalcazar@epn.edu.ec

numLayers = length(transferFunctions);
reluThresh = options.reluThresh;

N = size(dataX, 1);
vectorOnes = ones(N,1);
A = cell(1, numLayers);
Z = cell(1, numLayers);
Z{1} = dataX;
A{1} = [vectorOnes Z{1}];
for i = 2:numLayers
    Z{i} = A{i - 1}*W{i - 1}';
    % Transfer function values
    switch transferFunctions{i}
        case 'logsig'
            A{i} = [vectorOnes logsig(Z{i})];
        case 'relu'
            A{i} = [vectorOnes bsxfun(@max, Z{i}, reluThresh)];
        case 'tanh'
            A{i} = [vectorOnes tansig(Z{i})];
        case 'purelin'
            A{i} = [vectorOnes Z{i}];
        case 'softplus'
            A{i} = [vectorOnes log(exp(Z{i}) + 1)];
        case 'elu'
            A{i} = [vectorOnes elu(Z{i})];
        otherwise
            error('Invalid transfer function. Valid options are elu, softplus, relu, logsig, tanh, and purelin');
    end
end
return