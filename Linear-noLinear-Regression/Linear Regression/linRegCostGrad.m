function [cost, grad] = linRegCostGrad(dataX, dataY, B, lambda)
% Esta función permite calcular el ECM y su gradiente para un problema de
% regresión lineal
%
% Entradas
% dataX    - Matriz de [N, n+1] donde cada fila es un ejemplo de
%            entrenamiento
% dataY    - Vector de [N, 1] donde cada elemento es un valor real
% B        - Vector de [n + 1,1] que contiene los coeficientes de la
%            regresión lineal: B = [beta_0, beta_1,..., beta_n]
% lambda   - Escalar no negativo que controla el valor de los pesos durante
%            el entrenamiento de la regresión lineal
% Salidas
% cost     - Escalar que contiene el valor del error cuadrático medio
% grad     - Vector de [1, n + 1] que contiene los gradientes del ECM
%            con respecto a los coeficientes de la regresión lineal
%
% Marco E. Benalcázar
% marco.benalcazar@epn.edu.ec

% Número de ejemplos de entrenamiento
N = size(dataX, 1);

% Predicción de la regresión lineal
dataYHat = linRegTest(dataX, B);

% Cálculo del costo
ECM = (1/(2*N))*(dataY - dataYHat)'*(dataY - dataYHat);
regularization = (lambda/2)*sum(B(2:end).^2);
cost = ECM + regularization;

% Cálculo del gradiente del costo con respecto a B
grad = (-1/N)*[ones(N,1), dataX]'*(dataY - dataYHat);
grad(2:end) = grad(2:end) + lambda*B(2:end);
return