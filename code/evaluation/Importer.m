classdef Importer
    %IMPORTER Imports data from CSV and MAT/PNG
    %
    %   Uses a file path to a file and loads data to a cube of features and
    %   a matrix of labels.
    
    properties
    end
    
    methods(Static)
        
        function [labels, cube] = loadDataFrom(path)
            %LOADDATAFROM load data from a csv or mat/png file and return
            %             labels and features
            %
            %   The static function takes a path and checks whether it is a
            %   csv or a mat file. For a mat file the path to the
            %   corresponding png file is calculated. The files are loaded
            %   and reshaped to a (x, 1) matrix for the labels and a
            %   (x, 1, z) cube for the features.
            %
            %%  Input:
            %       path ... a path to a csv or mat/png file
            %
            %%  Output:
            %       labels . a 2-dimensional matrix of size (x, 1) filled
            %                with the labels of the corresponding instance 
            %                 in cube. If the instance is unlabeled, the
            %                 value of labels for this instance is zero.
            %       cube ... a 3-dimensional matrix of size (x, 1, z) where
            %                x is the number of instances and z the number
            %                of features.
            %
            % Version: 2016-11-29
            % Author: Tilman Krokotsch
            %%
            
            % if path to csv
            if (strcmp(path(end-2:end), 'csv') == 1)
                % read csv
                dataSet = dlmread(path, ';');
                % rearrange dimensions to (1, 3, 2)
                dataSet = permute(dataSet, [1, 3, 2]);
                % separate labels from features
                labels = dataSet(:, :, 1);
                cube = dataSet(:, :, 2:end);
            end
            
            % if path to mat
            if (strcmp(path(end-2:end), 'mat') == 1)
                % read features from mat
                loadedData = load(path, 'cube');
                cube = loadedData.cube;
                % reshape features to (x*y, 1, z)
                cube = reshape(cube, [], 1, size(cube, 3));
                % read labels from png
                labels = imread([path(1:end-3), 'png']);
                % reshape labels to (x*y, 1)
                labels = reshape(labels, [], 1, size(labels, 3));
            end
        end
        
    end
    
end

