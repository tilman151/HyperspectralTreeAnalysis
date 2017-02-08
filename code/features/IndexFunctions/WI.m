classdef WI < FeatureExtractor    
    %ContinuumRemoval continuum removal
    %
    %    WI is a wrapper class for wi and calculates the WI index
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
            features = wi(originalFeatures);
        end
        
        function str = toString(obj)
            str = class(obj);
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
    end
end
function [ result ] = wi( rawFeatures )
%WI Summary of this function goes here
%   Detailed explanation goes here

    r900 = rawFeatures(:,:,134);
    r970 = rawFeatures(:,:,152);
    result = r900./r970;
end

