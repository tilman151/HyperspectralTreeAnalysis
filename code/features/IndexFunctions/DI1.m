classdef DI1 < FeatureExtractor    
    %ContinuumRemoval continuum removal
    %
    %    DI1 is a wrapper class for di1 and calculates the DI1 index
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
        function features = extractFeatures(obj, originalFeatures)
            features = di1(originalFeatures);
        end
        
        function str = toString(obj)
            str = class(obj);
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
    end
end

function [ result ] = di1( rawFeatures )
%D1 Summary of this function goes here
%   Detailed explanation goes here
    r800 = rawFeatures(:,:,107);
    r550 = rawFeatures(:,:,107);
    result = r800 - r550;
end

