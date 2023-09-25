function [B, cost, iterations] = linRegTrain(dataX, dataY, typeOfLinearRegression, numIterations, varargin)
%% Esta funci�n permite entrenar una regresi�n lineal
%
% Entradas
% dataX      - Matriz [Nxn] donde cada fila es un vector
% dataY      - Vector [Nx1] donde cada elemento es el valor real asociado a su
%              vector en dataXTrain: dataY(i) es la salida del vector
%              dataX(i, :)
% typeOfLinearRegression    Indica el tipo de regresi�n lineal a ejecutar:
%                           - exact: Ejecuta la regresi�n usando la
%                             expresi�n B = inv(X'*X)*X'*Y
%                           - ridge: retorna el vector B que minimiza la
%                              funci�n:
%                              J(B) = MSE(y, f(x,B)) + (lambda/2)*Reg(B),
%                              Reg(B) = (b1)^2 +(b2)^2 + ... + (bn)^2 
%                           - lasso: retorna el vector B que minimiza la
%                              funci�n:
%                              J(B) = MSE(y, f(x,B)) + lambda*Reg(B),
%                              Reg(B) = |b1| + |b2| + ... + |bn| 
% numIterations - Entero que indica el n�mero m�ximo de iteraciones para el
%                 entrenamiento de la regresi�n log�stica
% Salida
% B          - B = [b0, b1, ..., bn]', es un vector [(n+1), 1] que contiene
%              los coeficientes de la regresi�n log�stica
% cost       - Vector con un n�mero de componentes igual a iterations
% iterations - Entero que indica el n�mero de iteraciones que se usaron
%              para entrenar la regresi�n log�stica
%
% Marco E. Benalc�zar, Ph.D.
% marco.benalcazar@epn.edu.ec

if nargin > 4
    lambda = varargin{1};
else
    lambda = 0;
end

%% Inicializaci�n del vector de coeficientes
[N,n] = size(dataX);
if strcmp(typeOfLinearRegression, 'exact')
    dataXp = [ones(N,1), dataX];
    B = dataXp'*dataXp\dataXp'*dataY;
    cost = NaN;
    iterations = NaN;
elseif strcmp(typeOfLinearRegression, 'ridge')
    BStart = zeros(n+1, 1);
    %% Configuraci�n del optimizador
    options = optimset('GradObj', 'on', 'MaxIter', numIterations);
    costFcn = @(t)linRegCostGrad(dataX, dataY, t, lambda);
    [B, cost, iterations] = fmincg(costFcn, BStart, options);
elseif strcmp(typeOfLinearRegression, 'lasso')
    BStart = zeros(n+1, 1);
    %% Configuraci�n del optimizador
    options = optimset('GradObj', 'on', 'MaxIter', numIterations);
    costFcn = @(t)lassoCostGrad(dataX, dataY, t, lambda);
    [B, cost, iterations] = fmincg(costFcn, BStart, options);
else
    error('Regresi�n lineal no v�lida. Debe seleccionar entre exact, ridge y lasso');
end

%% C�lculo del coeficiente de determinaci�n (R^2)
% C�lculo del coeficiente de determinaci�n
dataYHat = linRegTest(dataX, B);
SSres = sum( (dataY - dataYHat).^2 );
SStot = sum( (dataY - mean(dataY)).^2 );
R2 = 1 - (SSres/SStot);
fprintf('Coeficiente de determinaci�n: R^2 = %1.2f \n', R2);

%% C�lculo del ECM de entrenamiento
ETrain = sum( (dataY - dataYHat).^2 )/(2*N);
fprintf('ECM de entrenamiento = %3.2f \n', ETrain);
return