classdef GNDVI < FeatureExtractor    
    %ContinuumRemoval continuum removal
    %
    %    GNDVI is a wrapper class for gndvi and calculates the GNDVI   index
    %
    %% Methods:
    %    ContinuumRemoval. Constructor.
    %    extractFeatures . See documentation in superclass
    %                      TransformationFeatureExtractor.
    %                      Returns (width x height x numDim) cube 
    %                      with the extracted features of the  
    %                      (weight x height) input instances.
    %
    % Version: 2016-12-14
    % Author: Tuan Pham Minh

    methods
        function features = extractFeatures(obj, originalFeatures, ~, ~)
            features = gndvi(originalFeatures);
        end
        
        function str = toString(obj)
            str = class(obj);
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
    end
end
function [ result ] = gndvi( rawFeatures )
%GNDVI Summary of this function goes here
%   Detailed explanation goes here

    r780 = rawFeatures(:,:,102);
    r550 = rawFeatures(:,:,41);
    result = (r780-r550)./(r780-r550);
    
    isNanMask = isnan(result);
    result(isNanMask) = 0;
end

