clc;
close all;
clear all;
warning off all;
% Este programa es una aplicación de regresión usando redes neuronales feed
% forward y una comparación con regresión lineal
%
% Marco E. Benalcázar, Ph.D.
% marco.benalcazar@epn.edu.ec
addpath('Linear Regression');
addpath('Feed-forward ANN Regression');

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
sigma = cov (dataX);
R = corrcoef(dataX(1:3047,1), dataY);
%% División (aleatoria) de los datos en datos de entrenamiento y datos de testeo
% Datos de entrenamiento
NTrain = 1000;
[dummy,idx] = sort( rand(N, 1) );
dataXTrain = dataX( idx(1:NTrain), :);
dataYTrain = dataY( idx(1:NTrain) );
% Datos de testeo
dataXTest = dataX( idx((NTrain+1):end), :);
dataYTest = dataY( idx((NTrain+1):end) );

% Gráfico de los datos
figure;
plot(dataXTrain, dataYTrain, '*');
hold all;
plot(dataXTest, dataYTest, '*');
%plot(dataX, y, '.k');
hold off;
xlabel('x');
ylabel('y');

%% Regresión Lineal
% Entrenamiento de la regresión lineal
typeOfLinearRegression = 'exact';
numIterations = 0;
lambda = 0;
[B, cost, iterations] = linRegTrain(dataXTrain, dataYTrain, typeOfLinearRegression, numIterations, lambda);
if ~strcmp(typeOfLinearRegression, 'exact')
    figure;
    plot( 1:length(cost), cost );
    xlabel('Número de iteración');
    ylabel('Costo');
    title('Curva de entrenamiento de la regresión lineal');
end

% Testeo de la regresión lineal
dataYTestHat_lin = linRegTest(dataXTest, B);
NTest = N - NTrain;
ETest_lin = ( 1/(2*NTest) )*sum( (dataYTest - dataYTestHat_lin).^2 );

% Recta predicha por la regresión lineal
figure(1)
hold all;
plot(dataXTest, dataYTestHat_lin, 'm', 'Linewidth', 2);
hold off;

%% Cálculo del Error de Bayes
EBayes = (sigma^2)/2;

%% Regresión usando RNA Feed-Forward
% Arquitectura de la red%%%%%%%%%%%%
% Los siguientes valores son los que el diseñador debe modificar
n = size(dataXTrain, 2);
numNeuronsLayers = [n, 256, 128, 1];
% Nota: no se recomienda usar purelin como función
% de activación de las neuronas de las capas
% ocultas
% Opciones de función de activación:
% logsig
% tanh
% relu
% elu
% sofplus
% purelin (se recomienda su uso a la salida de una
% RNA que se usa para regresión)
transferFunctions{1} = 'none';
transferFunctions{2} = 'relu';
transferFunctions{3} = 'relu';
transferFunctions{4} = 'purelin';
options.lambda = 1e-3;
% Entrenamiento de la red neuronal
options.numIterations = 1000;
%%%%%%%%%%%%%%%%%%%%%%%%
options.reluThresh = 1e-2;
 W = ffnnRegTrain(dataXTrain, dataYTrain,...
     numNeuronsLayers, transferFunctions, options);
% Testeo de la red neuronal
[Z, A] = ffnnRegTest(dataXTest, W, transferFunctions, options);
dataYTestHat_net = A{end}(:, 2);
% Error de testeo de la red neuronal
ETest_net = ( 1/(2*NTest) )*sum( (dataYTest - dataYTestHat_net).^2 );

% Impresión de resultados
fprintf('Coeficiente de correlación = %1.2f \n', R(1,2));
fprintf('ECM de testeo regresión lineal = %3.2f \n', ETest_lin);
fprintf('ECM de testeo de la RNA = %3.2f \n', ETest_net);
fprintf('ECM de Bayes = %3.2f \n', mean(EBayes));

% Cálculo del número de pesos de la red
numWeights = 0;
for i = 2:length(numNeuronsLayers)
    numWeights = numWeights + ...
        numNeuronsLayers(i-1)*numNeuronsLayers(i) + ...
        numNeuronsLayers(i);
end
fprintf('\nNúmero de ejemplos de entrenamiento de la RNA = %d\n',NTrain);
fprintf('Número de pesos de la RNA = %d\n',numWeights);

% Curva predicha por la red neuronal
figure(1)
hold all;
plot(dataXTest, dataYTestHat_net, '.g', 'Linewidth', 2);
hold off;
legend('Datos de entrenamiento', 'Datos de testeo', 'Curva verdadera',...
    'Resultado regresión lineal', 'Resultado red neuronal artificial');
