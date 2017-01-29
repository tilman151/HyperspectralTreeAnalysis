classdef FeatureExtractionMerger < FeatureExtractor    
    %FeatureExtractionMerger merge multiple feature extraction strategies
    %   
    %   merge multiple feature extraction strategies
    %
    %% Properties:
    %    featureExtractors . the feature extractors which should eb used
    %
    %% Methods:
    %    ContinuumRemoval. Constructor.
    %                      featureExtractors . cell array of feature
    %                                          extractors
    %    toString ........ See documentation in superclass
    %                      FeatureExtractor.
    %    toShortString ... See documentation in superclass
    %                      FeatureExtractor.
    %    extractFeatures . See documentation in superclass
    %                      FeatureExtractor.
    %
    % Version: 2016-12-14
    % Author: Tuan Pham Minh
    
    properties
        featureExtractors;
    end

    methods
        
        function obj = FeatureExtractionMerger(featureExtractors)
            obj.featureExtractors = featureExtractors;
        end
        
        function str = toString(obj)
            str = 'FeatureExtractionMerger';
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
        
        function features = extractFeatures(obj, originalFeatures, ...
                                                 maskMap, samplesetPath)
            f = @(x) x.extractFeatures(originalFeatures, ...
                                       maskMap, samplesetPath);
            features = ...
                cellfun(f, obj.featureExtractors, 'uniformoutput', 0);
            features = cat(3, features{:});
        end
    end
    
end
