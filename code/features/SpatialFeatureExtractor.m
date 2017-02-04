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
            obj.extractionFunctions = {@extractMean, ...
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
            
            [y,x,numOriginalFeatures] = size(originalFeatures);
            
            numFeatures = numOriginalFeatures * numFeaturesToExtract;
            
            features = zeros(y,x,numFeatures);
            
            r = obj.radius;
            functions = obj.extractionFunctions;
            
            originalFeatures(maskMap<0) = nan; 
            if obj.multithreading
                for foIdx = 1:numOriginalFeatures
                    neighborhoodMatrix = ...
                        generateNeighborhoodMatrix( ...
                            originalFeatures(:,:,foIdx), ...
                            r);
                    
                    for efIdx = 1:numFeaturesToExtract
                        fIdx = (foIdx-1) * numFeaturesToExtract + efIdx;
                        extractionFunction = functions{efIdx};
                        f = extractionFunction(neighborhoodMatrix, r);
                        features(:,:,fIdx) = f;
                    end
                end
            else
                for ofIdx = 1:numOriginalFeatures
                    tic;
                    validPixel = maskMap ~= -1;
                    slice = originalFeatures(:,:,ofIdx);
                    for ix = 1:x
                        if mod(ix, 100) == 0
                            ix
                        end
                        for iy = 1:y
                            f = zeros(numFeaturesToExtract,1);
                            if validPixel(iy, ix)  
                                lx = max(1,ix - r);
                                rx = min(x,ix + r);
                                dy = max(1,iy - r);
                                uy = min(y,iy + r);

                                window = slice(dy:uy, lx:rx);
                                maskWindow = validPixel(dy:uy, lx:rx);
                                if sum(maskWindow(:)) > 0
                                    localY =  iy - dy + 1;
                                    localX =  ix - lx + 1;
                                    wx = (rx-lx+1);
                                    wy = (uy-dy+1);
                                    llx = max(1, r-(wx-localX)+1);
                                    lrx = llx + wx - 1;
                                    ldy = max(1, r-(wy-localY)+1);
                                    luy = ldy + wy - 1;
                                    nMatrix = nan(r*2+1);
                                    nMatrix(ldy:luy, llx:lrx) = window;
                                    nMatrix = ...
                                        reshape(nMatrix, ...
                                                1,1,numel(nMatrix));

                                    for efIdx = 1:numFeaturesToExtract
                                        extractionFunction = ...
                                            functions{efIdx};
                                        f(efIdx) = ...
                                            extractionFunction(nMatrix, r);
                                    end
                                end
                            end
                            lIdx = (ofIdx-1)*numFeaturesToExtract+1;
                            rIdx = ofIdx*numFeaturesToExtract;
                            features(iy,ix,lIdx:rIdx) = f;
                        end 
                    end
                    toc;
                end
            end
        end
    end
end

function neighborHoodMatrix = generateNeighborhoodMatrix(features, r)
    height = (2*r+1);
    neighborhoodSize = height^2;
    [y,x] = size(features);
    neighborHoodMatrix = zeros(y,x,neighborhoodSize);
    for nIdx = 1:neighborhoodSize
        shiftY = mod(nIdx-1, height) - r;
        shiftX = floor(nIdx-1/ height) - r;
        neighborHood = ...
            circshift(features, ...
                      [shiftY, shiftX]);
        
        invalidmask = ones(size(features));
        if shiftX < 0
            invalidmask(:,end+shiftX:end) = nan;
        elseif shiftX > 0
            invalidmask(:,1:shiftX) = nan;
        end
        if shiftY < 0
            invalidmask(end+shiftX:end,:) = nan;
        elseif shiftY > 0
            invalidmask(1:shiftX,:) = nan;
        end

        neighborHoodMatrix(:,:,nIdx) = neighborHood.*invalidmask;
    end
end

function functionName = extractFunctionName(functionHandle, keyword)
    functionName = func2str(functionHandle);
    if strcmp(keyword, functionName(1:numel(keyword)))
        functionName = functionName((numel(keyword)+1):end); 
    end
end

function result = extractMean(neighborhoodMatrix, ~)
    result = nanmean(neighborhoodMatrix,3);
end

function result = extractWeightedMean(neighborhoodMatrix, radius)
    width = 2*radius+1;
    localPosition = [radius + 1, radius+1];
    [x,y] = meshgrid(1:width);
    x=abs(x-localPosition(2));
    y=abs(y-localPosition(1));
    d = x+y;
    id = ((max(d(:))+1) - d);
    w = id/sum(id(:));
    w = reshape(w,1,1,numel(w));
    [sx,sy, ~] = size(neighborhoodMatrix);
    transformed = neighborhoodMatrix .* repmat(w,sx,sy,1);
    result = nansum(transformed, 3);
end

function result = extractVar(neighborhoodMatrix, ~)
    result = nanvar(neighborhoodMatrix,1,3);
end

function result = extractMax(neighborhoodMatrix, ~)
    result = nanmax(neighborhoodMatrix,[],3);
end

function result = extractMin(neighborhoodMatrix, ~)
    result = nanmin(neighborhoodMatrix,[],3);
end

function result = extractMedian(neighborhoodMatrix, ~)
    result = nanmedian(neighborhoodMatrix,3);
end
