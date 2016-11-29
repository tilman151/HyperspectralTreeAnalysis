classdef Importer
    %IMPORTER Imports data from CSV and MAT/PNG
    %
    %   Uses a file path to a file and loads data to a cube of features and
    %   a matrix of labels.
    
    properties
    end
    
    methods(Static)
        
        function [labels, cube] = loadDataFrom(path)
            if (strcmp(path(end-2:end), 'csv') == 1)
                dataSet = dlmread(path,';');
                dataSet = permute(dataSet, [1,3,2]);
                labels = dataSet(:,:,1);
                cube = dataSet(:,:,2:end);
            end
            
            if (strcmp(path(end-2:end), 'mat') == 1)
                loadedData = load(path, 'cube');
                cube = loadedData.cube;
                cube = reshape(cube, [], 1, size(cube, 3));
                labels = imread([path(1:end-3), 'png']);
                labels = reshape(labels, [], 1, size(labels, 3));
            end
        end
        
    end
    
end

