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
    
    % Get class names
    t = readtable(['../data/ftp-iff2.iff.fraunhofer.de/Baumarten/'...
                   'BaumartenIDsUnique.csv'], ...
                   'FileEncoding', 'ISO-8859-1');
    classNames = t.Name(classes);
    
    % Append class distribution
    classDistribution = sum(confMat, 2);
    namesAndDistribution = cellfun(@(name, dist) ...
        [name ' (' num2str(dist) ')'], ...
        classNames, num2cell(classDistribution), ...
        'UniformOutput', false);
    
    % Normalize confMat row-based
    sums = repmat(sum(confMat, 2), [1, length(classes)]);
    confMat = confMat./sums;
    
    % Flip confMat up-down for heatmap display
    confMat = flipud(confMat);
    
    % Plot confMat
    f = figure('Name', 'Confusion Matrix', 'NumberTitle', 'off');
    hm = HeatMap(confMat, false); % false prevents display of HeatMap obj
    hm.set('RowLabels', flipud(namesAndDistribution));
    hm.set('ColumnLabels', classNames);
    hm.set('Symmetric', false);
    hm.set('Colormap', jet);
    ax = hm.plot(f);
    colorbar(ax);
end

