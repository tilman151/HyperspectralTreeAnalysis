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
    %% Properties
    %       crossValParts .... Matrix of numbers of files for the test set
    %       filesPerClass .... Numbers of files ordered by contained class
    %       filePrefixFunc ... Function handle to the prefix function for
    %                          variable names
    %       k ................ Number of sets
    %       dataPath ......... Path to the data set
    %
    %% Methods
    %     CROSSVALIDATOR . The function receives a matrix of the numbers of
    %       files to use as test set, a list of the file numbers ordered by
    %       class in the file, the path to the data set and optional a
    %       prefix function. If no prefix function is supplied, the
    %       standard of the class is used.
    %
    %     GETTESTSET ... Returns the test set specified by the i_th
    %       column of crossValParts, containing labels and features.
    %
    %     GETTRAININGSET ... Returns the training set specified by the i_th
    %       column of crossValParts, containing labels and features.
    %
    %     LOADDATA ... Loads the variables specified by fileNums from the
    %       files labels.mat and features.mat in the specified path.
    %
    %     GETFIRST ... Returns the first element of a vector and [] if
    %       empty.
    %
    % Version: 2016-12-15
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
            % Check if index is in range
            if i < 0 || i > obj.k
                error('MATLAB:badsubscript',...
                      'Index exceeds matrix dimensions.');
            end
            
            %  Get the numbers of files for test set
            fileNums = obj.crossValParts(:, i);
            fileNums = fileNums(fileNums > 0);
            
            % Load test set
            [testLabels, testFeatures] = obj.loadData(fileNums);
            
        end
        
        function [testLabels, testFeatures] = getTrainingSet(obj, i)
            % Check if index is in range
            if i < 0 || i > obj.k
                error('MATLAB:badsubscript',...
                      'Index exceeds matrix dimensions.');
            end
            
            %  Get the numbers of files for training set
            fileNums = obj.crossValParts(:, i);
            fileNums = cellfun(@setdiff, obj.filesPerClass, ...
                               num2cell(fileNums), 'UniformOutput', false);
            fileNums = cellfun(@obj.getFirst, fileNums, ...
                               'UniformOutput', false);
            fileNums = [fileNums{:}];
            
            % Load training set
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
            % Calculate variable names from fileNums via prefixFunc
            fileNames = arrayfun(obj.filePrefixFunc, fileNums,...
                                 'UniformOutput', false);
            
            % Calculate sizes of all variables to be loaded
            fileSizes = cellfun(...
                   @(x)whos('-file', [obj.dataPath, 'features.mat'], x),...
                   fileNames);
            fileSizes = reshape([fileSizes.size], 3, []);
            
            % Calculate horizontal size of resulting matrix
            horSize = sum(fileSizes(2, :)) + size(fileSizes, 2)-1;
            
            % Initialize return values
            labels = zeros(fileSizes(1, 1), horSize)-1;
            features = zeros(fileSizes(1, 1), horSize, fileSizes(3, 1));
            
            % For each variable to be loaded
            xBegin = 1;
            for i = 1:size(fileNames, 1)
                % Load variables 
                tmpLabels = load([obj.dataPath, 'labels.mat'],...
                                 fileNames{i});
                tmpFeatures = load([obj.dataPath, 'features.mat'],...
                                   fileNames{i});
                               
                xEnd = xBegin + fileSizes(2, i)-1;
                
                % Copy loaded variables to return values
                labels(:, xBegin:xEnd) = ...
                                             tmpLabels.(fileNames{i});
                features(:, xBegin:xEnd, :) = ...
                                             tmpFeatures.(fileNames{i});
                                         
                xBegin = xEnd+2;
            end
            
        end
        
        function c = getFirst(~, c)
            if ~isempty(c)
                c = c(1);
            end
        end
        
    end
    
end

