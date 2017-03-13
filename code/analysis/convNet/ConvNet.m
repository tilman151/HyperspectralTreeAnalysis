classdef ConvNet < Classifier
%CONVNET convolutional neural net
%
%    Convolutional neural network that extracts training batches from a
%    given feature cube and learns on them for a given number of epochs.
%    The trained model can then be used for classification.
%    Make sure that matConvNet has been compiled by running `vl_compilenn`.
%
%% Properties:
%    opts ....... Options regarding training
%    net ........ Network structure
%    dimMeans ... Mean of training data for each input dimension
%    dimStds .... Standard deviation of training data for each input 
%                dimension
%    errorRates . List of error rates for training epochs
%
%% Methods:
%    ConvNet ....... Constructor. Can take Name, Value pair arguments 
%                    that change the internal parameters of the net.
%    toString ...... See documentation in superclass Classifier.
%    toShortString . See documentation in superclass Classifier.
%    trainOn ....... See documentation in superclass Classifier.
%    classifyOn .... See documentation in superclass Classifier.
%
% Version: 2017-02-15
% Author: Marianne Stecklina & Cornelius Styp von Rekowski
%

    properties
        % Options
        opts;
        
        % Error rates for training epochs
        errorRates;
    end
    
    properties(Hidden=true)
        % Network structure
        net;
        
        % Mean and standard deviation of the training data
        dimMeans;
        dimStds;
    end

    methods
        function obj = ConvNet(varargin)
            % Get default options and store them for later use in training
            obj.opts = setOpts(varargin{:});
        end

        function str = toString(obj)
            str = sprintf('ConvNet:\n\n');
            
            % Append string representation of whole options struct
            str = [str evalc('disp(obj.opts)')];
        end
        
        function str = toShortString(obj)
            str = 'ConvNet';
            
            % Append dropout rate
            str = [str '_dr' num2str(obj.opts.dropoutRate)];
            
            % Append pooling if used
            if obj.opts.doPooling
                str = [str '_pooling'];
            end
            
            % Append maximum number of epochs
            str = [str '_e' num2str(obj.opts.numEpochs)];
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Get logger
            logger = Logger.getLogger();
            
            % Create network structure
            logger.info('ConvNet', 'Create network structure');
            global NUMCLASSES;
            obj.net = createNet(...
                obj.opts.sampleSize, size(trainFeatureCube, 3), ...
                NUMCLASSES, obj.opts.filterSize, obj.opts.dropoutRate, ...
                obj.opts.doPooling);
            
            % Fill missing initial learning rate and weight decay values
            obj.net.layers = fillMissingInitialValues(obj.net.layers);
            
            % Preprocessing: normalize data to mean 0 and variance 1
            logger.info('ConvNet', 'Normalize data');
            [obj.dimMeans, obj.dimStds] = ...
                calculateDataStats(trainFeatureCube, trainLabelMap);
            trainFeatureCube = ...
                normalizeData(trainFeatureCube, obj.dimMeans, obj.dimStds);
            
            % Get shuffeled list of labeled pixels
            [labeled(1, :), labeled(2, :)] = find(trainLabelMap > 0);
            labeled = labeled(:, randperm(size(labeled, 2)));
            
            % Initialize error rate plot
            if obj.opts.plotErrorRates
                figure('Name', 'ConvNet Error Rates');
                axis([1, obj.opts.numEpochs, 0, 1]);
                errorLine = animatedline('Color', 'b');
                drawnow;
            end
            
            % Train network for the given number of epochs, unless for the
            % given window of epochs no significant improvement (defined by
            % the given error rate margin) has been found.
            % TODO: Load snapshot and start with epoch n
            state = [];
            minErrorRate = 1.0;
            minErrorEpoch = 0;
            obj.errorRates = zeros(1, obj.opts.numEpochs);
            for epoch = 1 : obj.opts.numEpochs
                logger.info('ConvNet', ['Epoch ' num2str(epoch) '/' ...
                    num2str(obj.opts.numEpochs)]);
                
                % Initialize state with momentum 0
                state = initStateMomentum(state, obj.net.layers);

                % Move CNN  to GPU as needed
                numGPUs = numel(obj.opts.gpus);
                [obj.net, state] = moveTo(numGPUs, obj.net, state, 'gpu');
                
                % Initialize result and error structures
                res = [];
                error = [];
                
                % Process data in batches
                numBatches = ceil(length(labeled) / obj.opts.batchSize);
                for batchIndex = 1 : numBatches
                    logger.debug('ConvNet', ['Batch ' ...
                        num2str(batchIndex) '/' num2str(numBatches)]);
                    
                    % Create batch from indices
                    logger.trace('ConvNet', 'Create batch');
                    batch = createBatch(labeled, batchIndex, ...
                        obj.opts.batchSize, obj.opts.sampleSize, ...
                        trainFeatureCube, trainLabelMap, true);
                    
                    % Move batch to GPU if possible
                    if numGPUs >= 1
                        batch.features = gpuArray(batch.features);
                    end
                    
                    % Set target
                    obj.net.layers{end}.class = batch.labels;
                    
                    % Train network
                    logger.trace('ConvNet', 'Train network');
                    res = vl_simplenn(obj.net, batch.features, 1, res, ...
                        'backPropDepth', obj.opts.backPropDepth, ...
                        'cudnn', obj.opts.cudnn);
                    
                    % Accumulate errors
                    logger.trace('ConvNet', 'Accumulate errors');
                    error = sum([error, [... 
                        sum(double(gather(res(end).x)));
                        reshape(obj.opts.errorFunction(obj.opts, ...
                            batch.labels, res), [], 1); ]], 2);
                    
                    % Accumulate gradients
                    batchSize = length(batch.labels);
                    logger.trace('ConvNet', 'Accumulate gradients');
                    [obj.net, res, state] = accumulateGradients(...
                        obj.net, res, state, obj.opts, batchSize, []);
                end
                
                % Save error rate
                curErrorRate = double(error(2)/size(labeled, 2));
                obj.errorRates(epoch) = curErrorRate;
                logger.info('ConvNet', ...
                    ['Error rate: ' num2str(curErrorRate)]);
                
                % Refresh error rate plot
                if obj.opts.plotErrorRates
                    addpoints(errorLine, epoch, curErrorRate);
                    drawnow;
                end
                
                % Move network and state back to CPU
                [obj.net, state] = moveTo(numGPUs, obj.net, state, 'cpu');
                
                if ~obj.opts.saveMomentum
                    state.momentum = [];
                end
                
                % Check if new best error rate has been found
                if curErrorRate <= ...
                        (minErrorRate - obj.opts.stoppingErrorMargin)
                    
                    minErrorRate = curErrorRate;
                    minErrorEpoch = epoch;
                end
                
                % Check if an improvement has been found within the defined
                % window of epochs
                if minErrorEpoch < (epoch - obj.opts.stoppingEpochWindow)
                    logger.info('ConvNet', ['Stopped training after ' ...
                        num2str(epoch) ' epochs, because no improvement'...
                        ' of at least ' ...
                        num2str(obj.opts.stoppingErrorMargin) ...
                        ' could be achieved within the last ' ...
                        num2str(obj.opts.stoppingEpochWindow) ' epochs']);
                    break;
                end
            end
        end 
        
        function predictedLabelMap = ...
                classifyOn(obj, evalFeatureCube, maskMap, ~)
            
            % Get logger
            logger = Logger.getLogger();
            
            % Preprocessing: normalize data to mean 0 and variance 1
            logger.info('ConvNet', 'Normalize data');
            evalFeatureCube = ...
                normalizeData(evalFeatureCube, obj.dimMeans, obj.dimStds);
            
            % Pad image so that every valid pixel has a neighborhood
            [evalFeatureCube, newMaskMap] = padImage(...
                evalFeatureCube, maskMap, (obj.opts.sampleSize - 1) / 2);
            
            % Move CNN  to GPU as needed
            numGPUs = numel(obj.opts.gpus);
            obj.net = moveTo(numGPUs, obj.net, [], 'gpu');
            
            % Initialize result structure and prediction list
            res = [];
            predictedLabelList = [];
            
            % Find unlabeled pixels
            [unlabeled(1, :), unlabeled(2, :)] = find(newMaskMap >= 0);
            
            % Process data in batches
            numBatches = ceil(length(unlabeled) / obj.opts.batchSize);
            for batchIndex = 1 : numBatches
                logger.debug('ConvNet', ['Batch ' ...
                    num2str(batchIndex) '/' num2str(numBatches)]);
                
                % Create batch from indices
                logger.debug('ConvNet', 'Create batch');
                batch = createBatch(unlabeled, batchIndex, ...
                    obj.opts.batchSize, obj.opts.sampleSize, ...
                    evalFeatureCube, newMaskMap, false);
                
                if ~isempty(batch.features)
                    % Move batch to GPU if possible
                    if numGPUs >= 1
                        batch.features = gpuArray(batch.features);
                    end

                    % Set target 
                    % (even though this might not be needed in prediction)
                    obj.net.layers{end}.class = batch.labels;

                    % Train network
                    logger.debug('ConvNet', 'Classify');
                    res = vl_simplenn(obj.net, batch.features, [], res, ...
                            'cudnn', obj.opts.cudnn);

                    % Get predictions from output layer
                    % (last layer is softmaxerror)
                    batchOutput = gather(res(end-1).x);
                    [~, ~, numClasses, curBatchSize] = size(batchOutput);
                    batchOutput = ...
                        reshape(batchOutput, numClasses, curBatchSize);
                    [~, predictedLabels] = max(batchOutput, [], 1);
                    
                    % Append predictions to overall list
                    predictedLabelList = ...
                        [predictedLabelList; predictedLabels'];
                end
            end
            
            % Rebuild map representation
            predictedLabelMap = rebuildMap(predictedLabelList, maskMap);
        end
        
    end
end


function layers = fillMissingInitialValues(layers)
    for i = 1:numel(layers)
        J = numel(layers{i}.weights);
        if ~isfield(layers{i}, 'learningRate')
            layers{i}.learningRate = ones(1, J);
        end
        if ~isfield(layers{i}, 'weightDecay')
            layers{i}.weightDecay = ones(1, J);
        end
    end
end


function state = initStateMomentum(state, layers)
    if isempty(state) || isempty(state.momentum)
        for i = 1:numel(layers)
            for j = 1:numel(layers{i}.weights)
                state.momentum{i}{j} = 0;
            end
        end
    end
end

function [net, state] = moveTo(numGPUs, net, state, destination)
    switch destination
        case 'gpu', moveop = @(x) gpuArray(x);
        case 'cpu', moveop = @(x) gather(x);
        otherwise, error('Unknown destination ''%s''.', destination);
    end

    if numGPUs >= 1
        % Get logger
        logger = Logger.getLogger();
        logger.debug('ConvNet', ['Move network to ' destination]);
        
        % Move network to destination
        net = vl_simplenn_move(net, destination);
        
        % Move state to destination
        if ~isempty(state)
            for i = 1:numel(state.momentum)
                for j = 1:numel(state.momentum{i})
                    state.momentum{i}{j} = moveop(state.momentum{i}{j});
                end
            end
        end
    end
end
