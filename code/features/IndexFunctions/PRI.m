classdef PRI < FeatureExtractor    
    %ContinuumRemoval continuum removal
    %
    %    PRI is a wrapper class for pri and calculates the PRI index
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
            features = pri(originalFeatures);
        end
        
        function str = toString(obj)
            str = class(obj);
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
    end
end
function [ result ] = pri( rawFeatures )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    r531 = rawFeatures(:,:,35);
    r570 = rawFeatures(:,:,46);
    result = (r531-r570)./(r531+r570);
    
    isNanMask = isnan(result);
    isInfMask = isinf(result);
    result(isNanMask | isInfMask) = 0;
end

