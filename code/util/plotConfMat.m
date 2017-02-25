function plotConfMat(confMat)
%PLOTCONFMAT Visualize the confusion matrix
%    
%    Visualize the confusion matrix as a color-coded heatmap.
%
%% Input:
%    confMat .. Confusion matrix or path to corresponding CSV file.
% 
% Version: 2017-02-25
% Author: Cornelius Styp von Rekowski
% 
    
    if ischar(confMat)
        % Path to confusion matrix is given
        confMat = csvread(confMat);
    end
    
    % Get indices of classes, that are actually represented
    classes = find(any(confMat > 0, 2));
    
    % Reduce confMat to relevant part
    confMat = confMat(classes, classes);
    
    % Normalize confMat row-based
    sums = repmat(sum(confMat, 2), [1, length(classes)]);
    confMat = confMat./sums;
    
    % Flip confMat up-down for heatmap display
    confMat = flipud(confMat);
    
    % Plot confMat
    hm = HeatMap(confMat, ...
        'RowLabels', flipud(classes), 'ColumnLabels', classes, ...
        'Symmetric', false, 'Colormap', 'jet');
    
    % Close HeatMap plot, because it cannot be modified
    close all hidden;
    
    % Plot heatmap into regular axis
    ax = hm.plot();
    colorbar(ax);
end

