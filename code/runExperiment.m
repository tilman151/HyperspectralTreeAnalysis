function runExperiment(configFilePath)
    %runExperiment Run a machine learning experiment
    %with the parameters defined in the configuration file

    %Add the code directory and all subdirectories to path
    addpath(genpath('./'));

    config = textread(configFilePath,'%s','whitespace','\n');
    cellfun(@eval,config);

    %TODO: Check config validity

    %Read csv file from dataSetPath
    dataSet = dlmread(dataSetPath,';');

    %Separate labels from features
    labels = dataSet(:,1);
    features = dataSet(:,2:end);

    % Order of the group labels for cinfusion matrix
    order = unique(labels);
    % Stratified k-fold cross-validation
    cp = cvpartition(labels,'k',k);

    % Curry function to bind order and classifier 
    func = @(trainFeatures,trainLabels,evalFeatures,evalLabels)...
        crossValFunc(trainFeatures,trainLabels,evalFeatures,evalLabels,...
        order,classifier);

    % Run cross validation
    cfMat = crossval(func,features,labels,'partition',cp);
    % compute confusion matrix over all cross validation instances
    cfMat = reshape(sum(cfMat),7,7)

end

function confMat = crossValFunc(trainFeatures,trainLabels,...
                                evalFeatures,evalLabels,...
                                order,classifier)
    %crossValFunc Function for the matlab cross validation.
    %Has to be curried to bind order and classifier beforehands

    %Learn classifier on trainFeatures/Labels and classify evalFeatures
    classifier = classifier.trainOn(trainFeatures,trainLabels);
    labels = classifier.classifyOn(evalFeatures);
    %Calculate confusion matrix
    confMat = confusionmat(evalLabels,labels,'order',order);
    
end
