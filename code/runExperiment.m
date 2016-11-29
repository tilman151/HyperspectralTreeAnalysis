function cfMat = runExperiment(configFilePath)
    %RUNEXPERIMENT Run a machine learning experiment
    %              with the parameters defined in the configuration file
    %
    %   The function takes a path to a configuration file and reads it. The
    %   statements in the file are executed to initialize the needed
    %   variables for the experiment.
    %   
    %   The data set is read via the Importer class. The classes of the
    %   labels are extracted and a cross validation partition is
    %   calculated. Afterwards cross validation is performed with the
    %   classifier specified in the config file to obtain a confusion
    %   matrix.
    %
    %%  Input:
    %       configFilePath ... a path to the config file file
    %
    %%  Output:
    %       cfMat ............ a confusion matrix yielded by cross
    %                          validation using the classifier and the data
    %                          set defined in the config file.
    %
    % Version: 2016-11-29
    % Author: Tilman Krokotsch
    %%
    
    % Add the code directory and all subdirectories to path
    addpath(genpath('./'));

    % Read and execute config file
    file = fopen(configFilePath);
    config = textscan(file, '%s', 'whitespace', '\n');
    cellfun(@eval, config{1});

    % TODO: Check config validity

    % Read csv or mat/png file from dataSetPath
    [labels, features] = Importer.loadDataFrom(dataSetPath);    

    % Order of the group labels for cinfusion matrix
    order = unique(labels);
    % Stratified k-fold cross-validation
    cp = cvpartition(labels, 'k', k);

    % Curry function to bind order and classifier 
    func = @(trainFeatures, trainLabels, evalFeatures, evalLabels)...
        crossValFunc(trainFeatures, trainLabels, evalFeatures,...
                     evalLabels, order, classifier);

    % Run cross validation
    cfMat = crossval(func, features, labels, 'partition', cp);
    % compute confusion matrix over all cross validation instances
    cfMat = reshape(sum(cfMat), size(order, 1), size(order, 1));

end

function confMat = crossValFunc(trainFeatures, trainLabels, ...
                                evalFeatures, evalLabels, ...
                                order, classifier)
    %CROSSVALFUNC Function for the matlab cross validation.
    %
    %             Has to be curried to bind order and classifier
    %             beforehands.
    %
    %%  Input:
    %       trainFeatures ... a 3-dimensional matrix of instances for the
    %                         classifier to be trained on
    %       trainLabels ..... a 2-dimensional matrix of lables of the
    %                         training instances
    %       evalFeatures .... a 3-dimensional matrix of instances for the
    %                         classifier to be evaluated on
    %       evalLabels ...... a 2-dimensional matrix of lables of the
    %                         evaluation instances
    %       order ........... order of the classes in the confusion
    %                         matrix
    %       classifier ...... classifier to be trained and evaluated
    %
    %%  Output:
    %       confMat ..........a confusion matrix yielded by evaluating the
    %                         classifier 
    %%

    %Learn classifier on trainFeatures/Labels and classify evalFeatures
    classifier = classifier.trainOn(trainFeatures, trainLabels);
    labels = classifier.classifyOn(evalFeatures);
    %Calculate confusion matrix
    confMat = confusionmat(evalLabels, labels, 'order', order);
    
end
