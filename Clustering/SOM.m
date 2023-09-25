function cluster = SOM(data, width, height, topology, epochs)

som = selforgmap([width, height], 'topologyFcn', topology, 'distanceFcn', 'linkdist');

x = data;
som = configure(som, x);
som.trainParam.epochs = epochs;
som = train(som, x);

cluster = vec2ind(som(x));

figure, plotsompos(som, x);

figure, plotsomnd(som);
end
