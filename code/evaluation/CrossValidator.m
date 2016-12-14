classdef CrossValidator
    %CROSSVALIDATOR Computes training and test data for cross validation
    %   
    %   This class receives a matrix of integers representing the X in the
    %   name "F2G9_X". Each row of the matrix represents therefor a
    %   variable in labels.mat and features.mat
    %
    %   One column of the matrix represents the files to be tested against
    %   in the cross validation. The remaining ones are used as training
    %   data.
    %
    %   The crossValidator loads the necessary data and stiches the data
    %   sets together to return a test and a training set.
    %
    % Version: 2016-12-09
    % Author: Tilman Krokotsch
    %%
    
    properties
        crossValParts;
        filesPerClass;
        filePrefixFunc;
        k;
        dataPath;
    end
    
    methods
        
        function obj = CrossValidator(varargin)
            obj.crossValParts = varargin{1};
            obj.filesPerClass = varargin{2};
            obj.k = size(obj.crossValParts, 2);
            obj.dataPath = varargin{3};
            
            if nargin > 3
                obj.filePrefixFunc = varargin{4};
            else
                obj.filePrefixFunc =@(x)obj.defaultPrefixFunc(x);
            end
        end
        
        function [testLabels, testFeatures] = getTestSet(obj, i)
            if i < 0 || i > obj.k
                error('MATLAB:badsubscript',...
                      'Index exceeds matrix dimensions.');
            end
            
            fileNums = obj.crossValParts(:, i);
            fileNums = fileNums(fileNums > 0);
            
            [testLabels, testFeatures] = obj.loadData(fileNums);
            
        end
        
        function [testLabels, testFeatures] = getTrainingSet(obj, i)
            if i < 0 || i > obj.k
                error('MATLAB:badsubscript',...
                      'Index exceeds matrix dimensions.');
            end
            
            fileNums = obj.crossValParts(:, i);
            fileNums = cellfun(@setdiff, obj.filesPerClass, ...
                               num2cell(fileNums), 'UniformOutput', false);
            fileNums = [fileNums{:}];
            
            [testLabels, testFeatures] = obj.loadData(fileNums);
            
        end
        
    end
    
    methods (Access = private)
        
        function fileName = defaultPrefixFunc(~, x)
            if x < 10
                fileName = ['F2G9_0', num2str(x)];
            else
                fileName = ['F2G9_', num2str(x)];
            end
        end
        
        function [labels, features] = loadData(obj, fileNums)
            fileNames = arrayfun(obj.filePrefixFunc, fileNums,...
                                 'UniformOutput', false);
                             
            fileSizes = cellfun(...
                   @(x)whos('-file', [obj.dataPath, 'features.mat'], x),...
                   fileNames);
            fileSizes = reshape([fileSizes.size], 3, []);
            
            horSize = sum(fileSizes(2, :)) + size(fileSizes, 2)-1;
            
            labels = zeros(fileSizes(1, 1), horSize)-1;
            features = zeros(fileSizes(1, 1), horSize, fileSizes(3, 1));
            
            xBegin = 1;
            for i = 1:size(fileNames, 1)
                tmpLabels = load([obj.dataPath, 'labels.mat'],...
                                 fileNames{i});
                tmpFeatures = load([obj.dataPath, 'features.mat'],...
                                   fileNames{i});
                               
                xEnd = xBegin + fileSizes(2, i)-1;
                
                labels(:, xBegin:xEnd) = ...
                                             tmpLabels.(fileNames{i});
                features(:, xBegin:xEnd, :) = ...
                                             tmpFeatures.(fileNames{i});
                                         
                xBegin = xEnd+2;
            end
        end
        
    end
    
end

