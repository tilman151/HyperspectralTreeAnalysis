clear;
load crossValParts.mat;
c = CrossValidator(crossValParts, filesPerClass, '../data/ftp-iff2.iff.fraunhofer.de/Data/Hyperspectral/400-1000/');

sizes = zeros(10, 2);
times = zeros(10,1);

for i = 1:10
    tic;
    [labels, features] = c.getTrainingSet(i);
    [labels, features] = c.getTestSet(i);
    times(i) = toc;
    sizes(i, :) = size(labels);
    clear('labels', 'features');
end