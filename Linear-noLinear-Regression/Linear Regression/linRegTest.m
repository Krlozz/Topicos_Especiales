function dataYHat = linRegTest(dataX, B)
% Esta funci�n predice la salida dataYHat usando una regresi�n lineal
%
% Entradas
% dataX    - Matriz de [N, n + 1] donde cada fila es un ejemplo de
%            entrenamiento
% B        - Vector de [n + 1,1] que contiene los coeficientes de la
%            regresi�n lineal: B = [beta_0, beta_1,..., beta_n]'
% Salida
% dataYHat - Vector de [N, 1] donde cada elemento es un valor real
%
% Marco E. Benalc�zar
% marco.benalcazar@epn.edu.ec

N = size(dataX,1);
dataYHat = [ones(N,1), dataX]*B;
return