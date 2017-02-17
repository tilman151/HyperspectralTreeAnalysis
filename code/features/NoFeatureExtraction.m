classdef NoFeatureExtraction < FeatureExtractor    
    %NoFeatureExtraction pretends to be a feature extractor without doing 
    %                    anything but returning the same name
    %
    %% Properties:
    %   baseFeatureExtractor . the feature extraction whose name should be
    %                          used
    %
    %% Methods:
    %    Indices ......... Constructor.
    %    toString ........ See documentation in superclass
    %                      FeatureExtractor.
    %    toShortString ... See documentation in superclass
    %                      FeatureExtractor.
    %    extractFeatures . See documentation in superclass
    %                      FeatureExtractor.
    %
    %
    % Version: 2017-02-17
    % Author: Tuan Pham Minh
    
    properties
        baseFeatureExtractor;
    end

    methods
        
        function obj = NoFeatureExtraction(baseFeatureExtractor)
            obj.baseFeatureExtractor = baseFeatureExtractor;
        end
        
        function str = toString(obj)
            str = obj.baseFeatureExtractor.toString();
        end
        
        function str = toShortString(obj)
            str = obj.baseFeatureExtractor.toShortString();
        end
        
        function features = extractFeatures(obj, originalFeatures, ~, ~)
        	features = originalFeatures;
        end
    end
    
end