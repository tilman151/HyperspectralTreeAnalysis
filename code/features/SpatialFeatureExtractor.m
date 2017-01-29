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
            str = ['SpatialFeatureExtractor {Radius: ' ...
                   num2str(obj.radius) ' Features: '];
            functionNames = ...
                cellfun(@(x) extractFunctionName(x, 'extract'), ...
                        obj.extractionFunctions, ...
                        'uniformoutput', 0);
                    
            functionNames = sort(functionNames);
            for fIdx = 1:numel(functionNames)
                if fIdx > 1
                    str = [str ', ' functionNames{fIdx}];
                else
                    str = [str functionNames{fIdx}];
                end
            end
            
            str = [str '}'];
        end
        
        function str = toShortString(obj)
            str = ['SpatialFeatureExtractor_' num2str(obj.radius)];
            functionNames = ...
                cellfun(@(x) extractFunctionName(x, 'extract'), ...
                        obj.extractionFunctions, ...
                        'uniformoutput', 0);
                    
            functionNames = sort(functionNames);
            for fIdx = 1:numel(functionNames)
                str = [str '_' functionNames{fIdx}];
            end
        end
        
        function features = extractFeatures(obj, originalFeatures, ...
                                                 maskMap, ~)
            numFeaturesToExtract = numel(obj.extractionFunctions);
            
            [x,y,numOriginalFeatures] = size(originalFeatures);
            
            numFeatures = numOriginalFeatures * numFeaturesToExtract;
            
            features = zeros(x,y,numFeatures);
            
            r = obj.radius;
            functions = obj.extractionFunctions;
            
            if obj.multithreading
               parfor fIdx = 1:numFeatures
                    validPixel = maskMap >= 0;
                    
                    sliceIdx = ceil(fIdx/numFeaturesToExtract);
                    slice = originalFeatures(:,:,sliceIdx);
                    functionIndex = ...
                        mod(fIdx - 1, numFeaturesToExtract) + 1;
                    extractionFunction = functions{functionIndex};
                    for ix = 1:x
                        for iy = 1:y
                            f = 0;
                            if validPixel(ix, iy) 
                                lx = max(1,ix - r);
                                rx = min(x,ix + r);
                                dy = max(1,iy - r);
                                uy = min(y,iy + r);

                                window = slice(lx:rx, dy:uy);
                                maskWindow = validPixel(lx:rx, dy:uy);
                                localPosition = [iy - dy + 1, ix - lx + 1];
                                f = extractionFunction(window, ...
                                                       localPosition, ...
                                                       maskWindow ...
                                                       );
                            end
                            features(ix,iy,fIdx) = f;
                        end 
                    end
                end
            else
                for fIdx = 1:numFeatures
                    validPixel = maskMap ~= -1;
                    sliceIdx = ceil(fIdx/numFeaturesToExtract);
                    slice = originalFeatures(:,:,sliceIdx);
                    functionIndex = ...
                        mod(fIdx - 1, numFeaturesToExtract) + 1;
                    extractionFunction = functions{functionIndex};
                    for ix = 1:x
                        for iy = 1:y
                            f = 0;
                            if validPixel(ix, iy) 
                                lx = max(1,ix - r);
                                rx = min(x,ix + r);
                                dy = max(1,iy - r);
                                uy = min(y,iy + r);

                                window = slice(lx:rx, dy:uy);
                                maskWindow = validPixel(lx:rx, dy:uy);
                                localPosition = [iy - dy + 1, ix - lx + 1];
                                f = extractionFunction(window, ...
                                                       localPosition, ...
                                                       maskWindow ...
                                                       );
                            end
                            features(ix,iy,fIdx) = f;
                        end 
                    end
                end
            end
        end
    end
end

function functionName = extractFunctionName(functionHandle, keyword)
    functionName = func2str(functionHandle);
    if strcmp(keyword, functionName(1:numel(keyword)))
        functionName = functionName((numel(keyword)+1):end); 
    end
end

function result = extractMean(featureWindow, ~, mask)
    result = mean(featureWindow(mask(:)));
end

function result = extractWeightedMean(featureWindow, localPosition, mask)
    [x,y] = meshgrid(1:(size(featureWindow,2)),1:size(featureWindow,1));
    x=abs(x-localPosition(2));
    y=abs(y-localPosition(1));
    d = x+y;
    id = ((max(d(:))+1) - d).* mask;
    w = id/sum(id(:));
    transformed = featureWindow .* w;
    result = sum(transformed(:));
end

function result = extractVar(featureWindow, ~, mask)
    result = var(featureWindow(mask(:)));
end

function result = extractMax(featureWindow, ~, mask)
    result = max(featureWindow(mask(:)));
end

function result = extractMin(featureWindow, ~, mask)
    result = min(featureWindow(mask(:)));
end

function result = extractMedian(featureWindow, ~, mask)
    result = median(featureWindow(mask(:)));
end
