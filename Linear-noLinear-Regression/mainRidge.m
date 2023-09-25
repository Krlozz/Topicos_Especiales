clc;
close all;
clear all;
warning off all;
% Este programa es una aplicación de la regresión lineal usando el dataset flores de isis
%
% Escuela Politécnica Nacional
% Marco E. Benalcázar, Ph.D.
% marco.benalcazar@epn.edu.ec

addpath('Linear Regression');
% rng('default') % For reproducibility

%% Carga de datos
% Cargamos los datos de entrenamiento
data = load('cancerReg.mat');

dataX = data.dataX(1:3047,:);
dataY = data.dataY(1:3047,:);

% Revolviendo los datos
N = size(dataX,1);
[dummy, rndIdx] = sort( rand(N,1) );
dataX = dataX(rndIdx, 1:21);
dataY = dataY(rndIdx);

%% División de los datos para entrenamiento y testeo
% Datos de entrenamiento
NTrain = 1000;
[dummy, randIdx] = sort( rand(N,1) );
dataXTrain = dataX(randIdx(1:NTrain), :);
dataYTrain = dataY(randIdx(1:NTrain));
% Datos de testeo
dataXTest = dataX(randIdx((NTrain+1):end), :);
dataYTest = dataY(randIdx((NTrain+1):end));

%% Entrenamiento del modelo
% Opciones de regresión lineal
% exact: B = inv(X'*X)*X'*Y
% ridge: Ridge linear regression
% lasso: LASSO linear regression
% OLS: ridge con lambda = 0
typeOfLinearRegression = 'ridge';  
numIterations = 50;
lambda = 0.01;
[B, cost, iterations] = linRegTrain(dataXTrain, dataYTrain, typeOfLinearRegression, numIterations, lambda);
if ~strcmp(typeOfLinearRegression, 'exact')
    figure;
    plot( 1:length(cost), cost );
    xlabel('Número de iteración');
    ylabel('Costo');
end

%% Testeo del modelo
NTest = N - NTrain;
dataYHat = linRegTest(dataXTest, B);
ETest = sum( (dataYTest - dataYHat).^2 )/(2*NTest);
fprintf('ECM de testeo = %3.2f \n', ETest);



