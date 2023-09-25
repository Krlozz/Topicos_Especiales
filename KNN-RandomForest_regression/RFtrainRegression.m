function [trainedModel, validationRMSE, RMSE] = RFtrainRegression(datos, minLeafSize, numlearners)

data = datos;
predictorNames = {'Berri1', 'Boyer', 'Br_beuf', 'CSC_C_teSainte_Catherine_', 'Maisonneuve_2', 'Maisonneuve_3', 'Notre_Dame', 'Parc', 'PierDup', 'Rachel_H_telDeVille', 'Rachel_Papineau', 'Ren__L_vesque', 'Saint_Antoine', 'Saint_Urbain', 'Totem_Laurier', 'University'};
predictors = data(:, predictorNames);
response = data.Viger;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a regression model
% This code specifies all the model options and trains the model.
template = templateTree(...
    'MinLeafSize', minLeafSize, ...
    'NumVariablesToSample', 'all');
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', numlearners, ...
    'Learners', template);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedModel.RequiredVariables = {'Berri1', 'Boyer', 'Br_beuf', 'CSC_C_teSainte_Catherine_', 'Maisonneuve_2', 'Maisonneuve_3', 'Notre_Dame', 'Parc', 'PierDup', 'Rachel_H_telDeVille', 'Rachel_Papineau', 'Ren__L_vesque', 'Saint_Antoine', 'Saint_Urbain', 'Totem_Laurier', 'University'};
trainedModel.RegressionEnsemble = regressionEnsemble;
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
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 5);

% Compute validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Compute validation RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));

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
