function spaReSSEnseUnitTest()
%SPARESSENSEUNITTEST Summary of this function goes here
%   Detailed explanation goes here
    
    featureMap = createFeatureMap();
    labelMap = createLabelMap();
    
    c = VisualizingClassifier();
    sparessense = SpaReSSEnse(c, 2, true, false);
    
    disp('Without enrichment');
    c.trainOn(featureMap, labelMap);
    
    disp('With enrichment');
    sparessense.trainOn(featureMap, labelMap);
end

function featureMap = createFeatureMap()
    featureMap(:,:,1) = [1 1 1 10 10; 1 1 1 1 10; 10 10 10 1 10; ...
        10 10 10 10 10; 10 10 10 2 2; 2 2 2 10 10; 0 0 0 0 0];
end

function labelMap = createLabelMap()
    labelMap =[1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; ...
               0 0 0 2 0; 0 0 0 0 0; -1 -1 -1 -1 -1];
end
