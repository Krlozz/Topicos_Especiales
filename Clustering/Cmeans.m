function [score, y_pred] = Cmeans(data, num_clusters, fuzziness)
options = [fuzziness, 100, 1e-5, 0];
[centers, U] = fcm(data, num_clusters, options);

[~, y_pred] = max(U);

scoreInd = silhouette(data, y_pred);
score = mean(scoreInd);

figure;
silhouette(data, y_pred);
end
