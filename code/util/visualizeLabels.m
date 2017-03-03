function visualizeLabels(labelMap, name)
%VISUALIZELABELS Visualize the 2-dimensional label map
%    
%    Show the label map as a color-coded image. Fill pixels are shown in
%    black, unlabeled pixels in white and labels are shown in colors.
% 
%% Input:
%    labelMap .. Matrix of labels with dimensions X x Y (being the image
%                dimensions). Entries are labels between 1 and the number 
%                of classes, while 0 can be used for unlabeled pixels and 
%                -1 for fill pixels.
%    name ...... [Optional] Name for the created figure.
% 
% Version: 2016-12-12
% Author: Cornelius Styp von Rekowski
%
    
    % Set default figure name, if no name is given
    if nargin < 2
        name = 'Label Matrix';
    end
    
    % Create color map
    % - fill pixels: black
    % - unlabeled: white
    % - labeled: colors
    global NUMCLASSES;
    colors = [0 0 0; 1 1 1; hsv(NUMCLASSES)];
    
    % Add 2 to the label map, because there might be -1 and 0 labels
    % that need to be transformed to valid color indices
    labelMap = labelMap + 2;
    
    % Show image and wait for it to be closed
    figure('Name', name);
    s = imshow(labelMap, colors, 'InitialMagnification', 'fit');
end
