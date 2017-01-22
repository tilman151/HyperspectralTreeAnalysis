classdef SpatialFeatureExtractor < FeatureExtractor    
    %SpatialFeatureExtractor extracts features inside of a window
    %
    %% Properties:
    %   
    %% Methods:
    %    ContinuumRemoval. Constructor.
    %                      radius ............ the radius in which the
    %                                          measures should be 
    %                                          calculated
    %                      useMultithreading . set to true to use a
    %                                          multithreaded implementation
    %    toString ........ See documentation in superclass
    %                      FeatureExtractor.
    %    toShortString ... See documentation in superclass
    %                      FeatureExtractor.
    %    extractFeatures . See documentation in superclass
    %                      TransformationFeatureExtractor.
    %                      Returns (width x height x numDim) cube 
    %                      with the extracted features of the  
    %                      (weight x height) input instances.
    %
    % Version: 2016-12-14
    % Author: Tuan Pham Minh
    
    properties
        radius;
        multithreading;
        extractionFunctions;
    end

    methods
        
        function obj = SpatialFeatureExtractor(radius, multithreading)
            obj.radius = radius;
            obj.multithreading = multithreading;
            obj.extractionFunctions = {@extractWeightedMean, ...
                                       @extractVar, ...
                                       @extractMin, ...
                                       @extractMax, ...
                                       };
%             obj.extractionFunctions = {@extractMean, ...
%                                        @extractWeightedMean, ...
%                                        @extractVar, ...
%                                        @extractMin, ...
%                                        @extractMax, ...
%                                        };
        end
        
        function str = toString(obj)
            str = 'SpatialFeatureExtractor';
        end
        
        function str = toShortString(obj)
            str = obj.toString();
        end
        
        function features = extractFeatures(obj, originalFeatures, ~)
            tic;
            numFeaturesToExtract = numel(obj.extractionFunctions);
            
            [x,y,numOriginalFeatures] = size(originalFeatures);
            
            numFeatures = numOriginalFeatures * numFeaturesToExtract;
            
            features = zeros(x,y,numFeatures);
            
            r = obj.radius;
            functions = obj.extractionFunctions;
            
            if obj.multithreading
               parfor fIdx = 1:numFeatures
                    sliceIdx = ceil(fIdx/numFeaturesToExtract);
                    slice = originalFeatures(:,:,sliceIdx);
                    functionIndex = ...
                        mod(fIdx - 1, numFeaturesToExtract) + 1;
                    extractionFunction = functions{functionIndex};
                    for ix = 1:x
                        for iy = 1:y
                            lx = max(1,ix - r);
                            rx = min(x,ix + r);
                            dy = max(1,iy - r);
                            uy = min(y,iy + r);

                            window = slice(lx:rx, dy:uy);
                            localPosition = [iy - dy + 1, ix - lx + 1];
                            features(ix,iy,fIdx) = ...
                                extractionFunction(window, localPosition);
                        end 
                    end
                end
            else
                for fIdx = 1:numFeatures
                    sliceIdx = ceil(fIdx/numFeaturesToExtract);
                    slice = originalFeatures(:,:,sliceIdx);
                    functionIndex = ...
                        mod(fIdx - 1, numFeaturesToExtract) + 1;
                    extractionFunction = functions{functionIndex};
                    for ix = 1:x
                        for iy = 1:y
                            lx = max(1,ix - r);
                            rx = min(x,ix + r);
                            dy = max(1,iy - r);
                            uy = min(y,iy + r);

                            window = slice(lx:rx, dy:uy);
                            localPosition = [iy - dy + 1, ix - lx + 1];
                            features(ix,iy,fIdx) = ...
                                extractionFunction(window, localPosition);
                        end 
                    end
                end
            end
            toc;
        end
    end
end

function result = extractMean(featureWindow, ~)
    result = mean(featureWindow(:));
end

function result = extractWeightedMean(featureWindow, localPosition)
    [x,y] = meshgrid(1:(size(featureWindow,2)),1:size(featureWindow,1));
    x=abs(x-localPosition(2));
    y=abs(y-localPosition(1));
    d = x+y;
    id = (max(d(:))+1) - d;
    w = id/sum(id(:));
    transformed = featureWindow .* w;
    result = sum(transformed(:));
end

function result = extractVar(featureWindow, ~)
    result = var(featureWindow(:));
end

function result = extractMax(featureWindow, ~)
    result = max(featureWindow(:));
end

function result = extractMin(featureWindow, ~)
    result = min(featureWindow(:));
end

function result = extractMedian(featureWindow, ~)
    result = median(featureWindow(:));
end
