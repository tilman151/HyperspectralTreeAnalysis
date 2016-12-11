classdef PrintClassifier < Classifier
    %PRINTCLASSIFIER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = trainOn(obj, trainFeatures, trainLabels)
            disp('Features');
            disp(trainFeatures);
            plotMap(trainFeatures);
            
            disp('Labels');
            disp(trainLabels);
            plotMap(trainLabels);
        end
        
        function labels = classifyOn(obj, evalFeatures)
            disp('classify');
        end
    end
    
end

function plotMap(labelMap)
    maxClass = max(max(labelMap));
    
    colors = [0 0 0; 1 1 1; hsv(maxClass)];
    
    s = imshow(labelMap + 2, colors, ...
        'InitialMagnification', 'fit');
    waitfor(s);
end

