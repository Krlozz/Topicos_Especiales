function [trainedModel, validationRMSE, RMSE] = SVM_train(datos, kernel, PolynomialOrder, KernelScale, numFolds)

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
data = datos;
predictorNames = {'Berri1', 'Boyer', 'Br_beuf', 'CSC_C_teSainte_Catherine_', 'Maisonneuve_2', 'Maisonneuve_3', 'Notre_Dame', 'Parc', 'PierDup', 'Rachel_H_telDeVille', 'Rachel_Papineau', 'Ren__L_vesque', 'Saint_Antoine', 'Saint_Urbain', 'Totem_Laurier', 'University'};
predictors = data(:, predictorNames);
response = data.Viger;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a regression model
% This code specifies all the model options and trains the model.
responseScale = iqr(response);
if ~isfinite(responseScale) || responseScale == 0.0
    responseScale = 1.0;
end
boxConstraint = responseScale/1.349;
epsilon = responseScale/13.49;
regressionSVM = fitrsvm(...
    predictors, ...
    response, ...
    'KernelFunction', kernel, ...
    'PolynomialOrder', PolynomialOrder, ...
    'KernelScale', KernelScale, ...
    'BoxConstraint', boxConstraint, ...
    'Epsilon', epsilon, ...
    'Standardize', true);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(regressionSVM, x);
trainedModel.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedModel.RequiredVariables = {'Berri1', 'Boyer', 'Br_beuf', 'CSC_C_teSainte_Catherine_', 'Maisonneuve_2', 'Maisonneuve_3', 'Notre_Dame', 'Parc', 'PierDup', 'Rachel_H_telDeVille', 'Rachel_Papineau', 'Ren__L_vesque', 'Saint_Antoine', 'Saint_Urbain', 'Totem_Laurier', 'University'};
trainedModel.RegressionSVM = regressionSVM;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2023a.';
trainedModel.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
data = datos;
predictorNames = {'Berri1', 'Boyer', 'Br_beuf', 'CSC_C_teSainte_Catherine_', 'Maisonneuve_2', 'Maisonneuve_3', 'Notre_Dame', 'Parc', 'PierDup', 'Rachel_H_telDeVille', 'Rachel_Papineau', 'Ren__L_vesque', 'Saint_Antoine', 'Saint_Urbain', 'Totem_Laurier', 'University'};
predictors = data(:, predictorNames);
response = data.Viger;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
KFolds = numFolds;
cvp = cvpartition(size(response, 1), 'KFold', KFolds);
% Initialize the predictions to the proper sizes
validationPredictions = response;
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;

    % Train a regression model
    % This code specifies all the model options and trains the model.
    responseScale = iqr(trainingResponse);
    if ~isfinite(responseScale) || responseScale == 0.0
        responseScale = 1.0;
    end
    boxConstraint = responseScale/1.349;
    epsilon = responseScale/13.49;
    regressionSVM = fitrsvm(...
        trainingPredictors, ...
        trainingResponse, ...
        'KernelFunction', 'linear', ...
        'PolynomialOrder', [], ...
        'KernelScale', 'auto', ...
        'BoxConstraint', boxConstraint, ...
        'Epsilon', epsilon, ...
        'Standardize', true);

    % Create the result struct with predict function
    svmPredictFcn = @(x) predict(regressionSVM, x);
    validationPredictFcn = @(x) svmPredictFcn(x);

    % Add additional fields to the result struct

    % Compute validation predictions
    validationPredictors = predictors(cvp.test(fold), :);
    foldPredictions = validationPredictFcn(validationPredictors);

    % Store predictions in the original order
    validationPredictions(cvp.test(fold), :) = foldPredictions;
end

% Compute validation RMSE
isNotMissing = ~isnan(validationPredictions) & ~isnan(response);
validationRMSE = sqrt(nansum(( validationPredictions - response ).^2) / numel(response(isNotMissing) ));


yfit = trainedModel.predictFcn(predictors);
RMSE = rmse(yfit, response);

x = yfit;  
y = [response, yfit];
figure;
title(['RMSE: ', num2str(RMSE)]);
plot(x, y, '.');
xlabel('Predicted Values');
ylabel('Actual Values');
legend('Actual', 'Predicted');
