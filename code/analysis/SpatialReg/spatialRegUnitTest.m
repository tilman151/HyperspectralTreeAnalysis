function spatialRegUnitTest()
%SPATIALREGUNITTEST Summary of this function goes here
%   Detailed explanation goes here
    
    featureCube = createFeatureCube();
    labelMap = createLabelMap();
    
    c = VisualizingClassifier();
    spatialReg = SpatialReg(c, 2, true, false);
    
    disp('Without enrichment');
    c.trainOn(featureCube, labelMap);
    
    disp('With enrichment');
    spatialReg.trainOn(featureCube, labelMap);
end

function featureCube = createFeatureCube()
    featureCube(:,:,1) = [1 1 1 10 10; 1 1 1 1 10; 10 10 10 1 10; ...
        10 10 10 10 10; 10 10 10 2 2; 2 2 2 10 10; 0 0 0 0 0];
end

function labelMap = createLabelMap()
    labelMap =[1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; ...
               0 0 0 2 0; 0 0 0 0 0; -1 -1 -1 -1 -1];
end
