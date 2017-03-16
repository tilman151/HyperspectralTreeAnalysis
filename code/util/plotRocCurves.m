function  plotRocCurves( confMats, names )
%PLOTROCCURVES generate 
%    
%    Visualize the ROC points for each confMat for each class separately
%
%% Input:
%    confMats .. CxCxN matrix with C being the number of classes and N the 
%                number of confusion matrices.
%    names ..... a cell array of strings with N elements which will be used
%                for the legend
% 
% Version: 2017-03-15
% Author: Tuan Pham Minh
% 

[numClasses, ~, numConfMats] = size(confMats);
sumAllConfMats = sum(confMats, 3);
hasInstances = (sum(sumAllConfMats,1) + sum(sumAllConfMats,2)') > 0;

classes = 1:numClasses;
classes(~hasInstances) = [];
numClasses = numel(classes);

colors = hsv(numConfMats);

for c = classes
    title = ['class: ' num2str(c)];
    figure('name', title);
    ylabel('TPR');
    xlabel('FPR');
    hold on;
    
    tpr = zeros(numConfMats,1);
    fpr = zeros(numConfMats,1);
    
    for confIdx = 1:numConfMats
        confMat = confMats(:,:,confIdx);
        tp = confMat(c,c);
        fp = sum(confMat(:,c)) - tp;
        fn = sum(confMat(c,:)) - tp;
        tn = sum(confMat(:)) - tp - fp - fn;
        cp = tp + fn;
        cn = fp + tn;
        tpr(confIdx) = tp/cp;
        fpr(confIdx) = fp/cn;
        
        scatter(fpr(confIdx),tpr(confIdx), 30, colors(confIdx,:), 'filled');
    end
    
    plot([0,1],[0,1], 'black');
    
    legend([names 'baseline'], 'location', 'eastoutside');
    
    
    ylim([0,1]);
    xlim([0,1]);
end

end

