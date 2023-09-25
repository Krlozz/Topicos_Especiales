function [score, y_pred] = Kmeans(data, num_clusters)
[y_pred, centers] = kmeans(data, num_clusters);

scoreInd = silhouette(data, y_pred);
score = mean(scoreInd);

figure;
silhouette(data, y_pred);
end