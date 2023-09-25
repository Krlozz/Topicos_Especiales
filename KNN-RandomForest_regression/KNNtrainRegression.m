function [trainedModel, RMSE] = KNNtrainRegression(trainingData, distancia, vecinos)
data = trainingData;

predictorNames = {'Berri1', 'Boyer', 'Br_beuf', 'CSC_C_teSainte_Catherine_', 'Maisonneuve_2', 'Maisonneuve_3', 'Notre_Dame', 'Parc', 'PierDup', 'Rachel_H_telDeVille', 'Rachel_Papineau', 'Ren__L_vesque', 'Saint_Antoine', 'Saint_Urbain', 'Totem_Laurier', 'University'};

predictors = data(:, predictorNames);
response = data.Viger;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];



knntrainedModel = fitcknn(predictors, response, 'NumNeighbors', vecinos, 'Distance', distancia,'Standardize', true);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(knntrainedModel, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedModel.KnntrainedModel = knntrainedModel;

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