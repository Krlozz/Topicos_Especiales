function [B, cost, iterations] = linRegTrain(dataX, dataY, typeOfLinearRegression, numIterations, varargin)
%% Esta función permite entrenar una regresión lineal
%
% Entradas
% dataX      - Matriz [Nxn] donde cada fila es un vector
% dataY      - Vector [Nx1] donde cada elemento es el valor real asociado a su
%              vector en dataXTrain: dataY(i) es la salida del vector
%              dataX(i, :)
% typeOfLinearRegression    Indica el tipo de regresión lineal a ejecutar:
%                           - exact: Ejecuta la regresión usando la
%                             expresión B = inv(X'*X)*X'*Y
%                           - ridge: retorna el vector B que minimiza la
%                              función:
%                              J(B) = MSE(y, f(x,B)) + (lambda/2)*Reg(B),
%                              Reg(B) = (b1)^2 +(b2)^2 + ... + (bn)^2 
%                           - lasso: retorna el vector B que minimiza la
%                              función:
%                              J(B) = MSE(y, f(x,B)) + lambda*Reg(B),
%                              Reg(B) = |b1| + |b2| + ... + |bn| 
% numIterations - Entero que indica el número máximo de iteraciones para el
%                 entrenamiento de la regresión logística
% Salida
% B          - B = [b0, b1, ..., bn]', es un vector [(n+1), 1] que contiene
%              los coeficientes de la regresión logística
% cost       - Vector con un número de componentes igual a iterations
% iterations - Entero que indica el número de iteraciones que se usaron
%              para entrenar la regresión logística
%
% Marco E. Benalcázar, Ph.D.
% marco.benalcazar@epn.edu.ec

if nargin > 4
    lambda = varargin{1};
else
    lambda = 0;
end

%% Inicialización del vector de coeficientes
[N,n] = size(dataX);
if strcmp(typeOfLinearRegression, 'exact')
    dataXp = [ones(N,1), dataX];
    B = dataXp'*dataXp\dataXp'*dataY;
    cost = NaN;
    iterations = NaN;
elseif strcmp(typeOfLinearRegression, 'ridge')
    BStart = zeros(n+1, 1);
    %% Configuración del optimizador
    options = optimset('GradObj', 'on', 'MaxIter', numIterations);
    costFcn = @(t)linRegCostGrad(dataX, dataY, t, lambda);
    [B, cost, iterations] = fmincg(costFcn, BStart, options);
elseif strcmp(typeOfLinearRegression, 'lasso')
    BStart = zeros(n+1, 1);
    %% Configuración del optimizador
    options = optimset('GradObj', 'on', 'MaxIter', numIterations);
    costFcn = @(t)lassoCostGrad(dataX, dataY, t, lambda);
    [B, cost, iterations] = fmincg(costFcn, BStart, options);
else
    error('Regresión lineal no válida. Debe seleccionar entre exact, ridge y lasso');
end

%% Cálculo del coeficiente de determinación (R^2)
% Cálculo del coeficiente de determinación
dataYHat = linRegTest(dataX, B);
SSres = sum( (dataY - dataYHat).^2 );
SStot = sum( (dataY - mean(dataY)).^2 );
R2 = 1 - (SSres/SStot);
fprintf('Coeficiente de determinación: R^2 = %1.2f \n', R2);

%% Cálculo del ECM de entrenamiento
ETrain = sum( (dataY - dataYHat).^2 )/(2*N);
fprintf('ECM de entrenamiento = %3.2f \n', ETrain);
return