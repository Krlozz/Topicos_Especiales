function [cost, grad] = linRegCostGrad(dataX, dataY, B, lambda)
% Esta funci�n permite calcular el ECM y su gradiente para un problema de
% regresi�n lineal
%
% Entradas
% dataX    - Matriz de [N, n+1] donde cada fila es un ejemplo de
%            entrenamiento
% dataY    - Vector de [N, 1] donde cada elemento es un valor real
% B        - Vector de [n + 1,1] que contiene los coeficientes de la
%            regresi�n lineal: B = [beta_0, beta_1,..., beta_n]
% lambda   - Escalar no negativo que controla el valor de los pesos durante
%            el entrenamiento de la regresi�n lineal
% Salidas
% cost     - Escalar que contiene el valor del error cuadr�tico medio
% grad     - Vector de [1, n + 1] que contiene los gradientes del ECM
%            con respecto a los coeficientes de la regresi�n lineal
%
% Marco E. Benalc�zar
% marco.benalcazar@epn.edu.ec

% N�mero de ejemplos de entrenamiento
N = size(dataX, 1);

% Predicci�n de la regresi�n lineal
dataYHat = linRegTest(dataX, B);

% C�lculo del costo
ECM = (1/(2*N))*(dataY - dataYHat)'*(dataY - dataYHat);
regularization = (lambda/2)*sum(B(2:end).^2);
cost = ECM + regularization;

% C�lculo del gradiente del costo con respecto a B
grad = (-1/N)*[ones(N,1), dataX]'*(dataY - dataYHat);
grad(2:end) = grad(2:end) + lambda*B(2:end);
return