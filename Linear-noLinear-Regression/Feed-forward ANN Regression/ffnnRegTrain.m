function W = ffnnRegTrain(dataX, dataY, numNeuronsLayers, transferFunctions, metaParameters)
%% This function trains a NEURAL NETWORK FOR REGRESSION by minimizing
%  the mean square function
%
% Inputs:
% dataX                   [N n] matrix, where each row contains an observation
%                         X = (x_1, x_2,...,x_n)
%
% dataY                   [N 1] vector, where each row contains an output
%
% numNeuronsLayers        [1 L] vector [#_1, #_2,..., #_L], where #_1
%                         denotes the size of the input layer, #_2 denotes
%                         the size of the first hidden layer, #_3 denotes
%                         the size of the second hidden layer, and so on, and
%                         #_L = 1 denotes the size of the output layer
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
% metaParameters         structure containing additional settings for the
%                        neural network (e.g., rectified linear unit
%                        threshold, lambda, number of iterations, etc.)
%
% Escuela Politecnica Nacional
% Advanced Machine Learning
% Marco E. Benalcázar Palacios
% marco.benalcazar@epn.edu.ec

fprintf('Training an artificial neural network\n');

% Initializing the Neural Network Parameters Randomly
mean = 0;
sigma = 0.01;
initialTheta = [];
for i = 2:length(numNeuronsLayers)   
    W = normrnd(mean, sigma, numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1);
    initialTheta = [initialTheta; W(:)];
end

% Unrolling parameters
options = optimset('MaxIter', metaParameters.numIterations);
costFunction = @(t) ffnnRegCostGrad(dataX, dataY,...
                                    numNeuronsLayers,...
                                    t,...
                                    transferFunctions,...
                                    metaParameters);
                                    
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[theta, cost, iterations] = fmincg(costFunction, initialTheta, options);

% Plotting the error curve
figure;
plot(1:length(cost),cost);
xlabel('Iteration');
ylabel('Cost');
title('Curva de entrenamiento de la red neuronal artificial');
grid on;
drawnow;
% Reshaping the weight matrices
numLayers = length(numNeuronsLayers);
endPoint = 0;
W = cell(1, numLayers - 1);
for i = 2:numLayers
    numRows = numNeuronsLayers(i);
    numCols = numNeuronsLayers(i - 1) + 1;    
    numWeights = numRows*numCols;
    startPoint = endPoint + 1;
    endPoint = endPoint + numWeights;
    W{i-1} = reshape(theta(startPoint:endPoint), numRows, numCols);
end  

% Computing the training error
[dummyVar, A] = ffnnRegTest(dataX, W, transferFunctions, metaParameters);
dataYHat = A{end}(:,2);
ETrain = (1/(2*length(dataY)))*sum( (dataYHat - dataY).^2 );
fprintf('Training MSE of the NEURAL NETWORK: %3.2f \n\n', ETrain);
end