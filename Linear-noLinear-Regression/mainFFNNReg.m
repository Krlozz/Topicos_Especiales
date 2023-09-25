clc;
close all;
clear all;
warning off all;
% Este programa es una aplicaci�n de regresi�n usando redes neuronales feed
% forward y una comparaci�n con regresi�n lineal
%
% Marco E. Benalc�zar, Ph.D.
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
%% Divisi�n (aleatoria) de los datos en datos de entrenamiento y datos de testeo
% Datos de entrenamiento
NTrain = 1000;
[dummy,idx] = sort( rand(N, 1) );
dataXTrain = dataX( idx(1:NTrain), :);
dataYTrain = dataY( idx(1:NTrain) );
% Datos de testeo
dataXTest = dataX( idx((NTrain+1):end), :);
dataYTest = dataY( idx((NTrain+1):end) );

% Gr�fico de los datos
figure;
plot(dataXTrain, dataYTrain, '*');
hold all;
plot(dataXTest, dataYTest, '*');
%plot(dataX, y, '.k');
hold off;
xlabel('x');
ylabel('y');

%% Regresi�n Lineal
% Entrenamiento de la regresi�n lineal
typeOfLinearRegression = 'exact';
numIterations = 0;
lambda = 0;
[B, cost, iterations] = linRegTrain(dataXTrain, dataYTrain, typeOfLinearRegression, numIterations, lambda);
if ~strcmp(typeOfLinearRegression, 'exact')
    figure;
    plot( 1:length(cost), cost );
    xlabel('N�mero de iteraci�n');
    ylabel('Costo');
    title('Curva de entrenamiento de la regresi�n lineal');
end

% Testeo de la regresi�n lineal
dataYTestHat_lin = linRegTest(dataXTest, B);
NTest = N - NTrain;
ETest_lin = ( 1/(2*NTest) )*sum( (dataYTest - dataYTestHat_lin).^2 );

% Recta predicha por la regresi�n lineal
figure(1)
hold all;
plot(dataXTest, dataYTestHat_lin, 'm', 'Linewidth', 2);
hold off;

%% C�lculo del Error de Bayes
EBayes = (sigma^2)/2;

%% Regresi�n usando RNA Feed-Forward
% Arquitectura de la red%%%%%%%%%%%%
% Los siguientes valores son los que el dise�ador debe modificar
n = size(dataXTrain, 2);
numNeuronsLayers = [n, 256, 128, 1];
% Nota: no se recomienda usar purelin como funci�n
% de activaci�n de las neuronas de las capas
% ocultas
% Opciones de funci�n de activaci�n:
% logsig
% tanh
% relu
% elu
% sofplus
% purelin (se recomienda su uso a la salida de una
% RNA que se usa para regresi�n)
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

% Impresi�n de resultados
fprintf('Coeficiente de correlaci�n = %1.2f \n', R(1,2));
fprintf('ECM de testeo regresi�n lineal = %3.2f \n', ETest_lin);
fprintf('ECM de testeo de la RNA = %3.2f \n', ETest_net);
fprintf('ECM de Bayes = %3.2f \n', mean(EBayes));

% C�lculo del n�mero de pesos de la red
numWeights = 0;
for i = 2:length(numNeuronsLayers)
    numWeights = numWeights + ...
        numNeuronsLayers(i-1)*numNeuronsLayers(i) + ...
        numNeuronsLayers(i);
end
fprintf('\nN�mero de ejemplos de entrenamiento de la RNA = %d\n',NTrain);
fprintf('N�mero de pesos de la RNA = %d\n',numWeights);

% Curva predicha por la red neuronal
figure(1)
hold all;
plot(dataXTest, dataYTestHat_net, '.g', 'Linewidth', 2);
hold off;
legend('Datos de entrenamiento', 'Datos de testeo', 'Curva verdadera',...
    'Resultado regresi�n lineal', 'Resultado red neuronal artificial');
