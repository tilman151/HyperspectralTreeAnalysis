function plotBoxPlot( resultMatrix, names, yLabelName, ylimits, exportPath, newSize)
%PLOTBOXPLOT Summary of this function goes here
%   Detailed explanation goes here
    
    figureHandle = figure();
    axisHandle = gca;
    hold on;
    boxplotHandle = boxplot(resultMatrix,'jitter',0,'whisker',inf);
    normalizeLineObject(boxplotHandle);
    xlabel('strategies');
	ylabel(yLabelName);
    names = cellfun(@(s)strrep(s,'\n', '\newline'), names, ...
        'UniformOutput', 0);
    set(axisHandle,'xTickLabels',names);
    set(axisHandle,'fontsize', 12, 'fontweight', 'bold');
    set(axisHandle,'TickLabelInterpreter', 'tex');
    set(axisHandle,'XTickLabelRotation', 90);
    normalizeAxis(axisHandle);

    if numel(ylimits) == 2
       ylim(ylimits); 
    end
    
    if exist('newSize', 'var') && ~isempty(newSize)
        curPos = get(figureHandle,'position');
        newPos = [curPos(1:2) newSize];
        newPos(2) = newPos(2) - newPos(4) + curPos(4) ;
        set(figureHandle,'position', newPos);
    end
    
    if exist('exportPath', 'var') && ~isempty(exportPath)
       [directoryPath, filename, fileEnding] = fileparts(exportPath);
       exportFigure(figureHandle, directoryPath, [filename fileEnding]); 
    end
end

