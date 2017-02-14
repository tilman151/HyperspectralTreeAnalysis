classdef ConvNet < Classifier
%CONVNET convolutional neural net
%
%    Convolutional neural network that extracts training batches from a
%    given feature cube and learns on them for a given number of epochs.
%    The trained model can then be used for classification.
%    Make sure that matConvNet has been compiled by running `vl_compilenn`.
%
%% Properties:
%    opts .. Options regarding training
%    net ... Network structure
%    mean .. Mean of training data for each input dimension
%    std ... Standard deviation of training data for each input dimension
%
%% Methods:
%    ConvNet ....... Constructor. Can take Name, Value pair arguments 
%                    that change the internal parameters of the net.
%    toString ...... See documentation in superclass Classifier.
%    toShortString . See documentation in superclass Classifier.
%    trainOn ....... See documentation in superclass Classifier.
%    classifyOn .... See documentation in superclass Classifier.
%
% Version: 2017-02-13
% Author: Marianne Stecklina & Cornelius Styp von Rekowski
%

    properties
        % Options
        opts;
        
        % Network structure
        net;
        
        % Mean and standard deviation of the training data
        mean;
        std;
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
        
        function str = toShortString(~)
            str = 'ConvNet';
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Get logger
            logger = Logger.getLogger();
            
            % Create network structure
            logger.info('ConvNet', 'Create network structure');
            global NUMCLASSES;
            obj.net = createNet(obj.opts.sampleSize, numDim, NUMCLASSES,...
                obj.opts.filterSize);
            
            % Fill missing initial learning rate and weight decay values
            obj.net.layers = fillMissingInitialValues(obj.net.layers);
            
            % Preprocessing: normalize data to mean 0 and variance 1
            logger.info('ConvNet', 'Normalize data');
            [trainFeatureCube, obj.mean, obj.std] = ...
                normalizeData(trainFeatureCube, trainLabelMap);
            
            % Get shuffeled list of labeled pixels
            [labeled(1, :), labeled(2, :)] = find(trainLabelMap > 0);
            labeled = labeled(:, randperm(size(labeled, 2)));
            
            % Train network for the given number of epochs
            % TODO: Load snapshot and start with epoch n
            state = [];
            for epoch = 1 : obj.opts.numEpochs
                logger.info('ConvNet', ['Epoch ' num2str(epoch) '/' ...
                    num2str(obj.opts.numEpochs)]);
                
                % Initialize state with momentum 0
                state = initStateMomentum(state, obj.net.layers);

                % Move CNN  to GPU as needed
                numGPUs = numel(obj.opts.gpus);
                [obj.net, state] = moveTo(numGPUs, obj.net, state, 'gpu');
                
                % Initialize result and error arrays
                res = [];
                error = [];
                
                % Process data in batches
                numBatches = ceil(length(labeled) / obj.opts.batchSize);
                for batchIndex = 1 : numBatches
                    logger.info('ConvNet', ['Batch ' ...
                        num2str(batchIndex) '/' num2str(numBatches)]);
                    
                    % Create batch from indices
                    logger.debug('ConvNet', 'Create batch');
                    batch = createBatch(labeled, batchIndex, ...
                        obj.opts.batchSize, obj.opts.sampleSize, ...
                        trainFeatureCube, trainLabelMap);
                    
                    % Move batch to GPU if possible
                    if numGPUs >= 1
                        batch.data = gpuArray(batch.data) ;
                    end
                    
                    % Set target
                    obj.net.layers{end}.class = batch.labels;
                    
                    % Train network
                    logger.debug('ConvNet', 'Train network');
                    res = vl_simplenn(obj.net, batch.features, 1, res, ...
                        'backPropDepth', obj.opts.backPropDepth, ...
                        'cudnn', obj.opts.cudnn);
                    
                    % Accumulate errors
                    logger.debug('ConvNet', 'Accumulate errors');
                    error = sum([error, [... 
                        sum(double(gather(res(end).x)));
                        reshape(obj.opts.errorFunction(obj.opts, ...
                            batch.labels, res), [], 1); ]], 2);

                    % Accumulate gradients
                    % TODO: change batchSize to real size of batch which
                    %       might be smaller
                    logger.debug('ConvNet', 'Accumulate gradients');
                    [obj.net, res, state] = accumulateGradients(...
                        obj.net, res, state, obj.opts, ...
                        obj.opts.batchSize, []) ;
                end
                
                % Move network and state back to CPU
                [obj.net, state] = moveTo(numGPUs, obj.net, state, 'cpu');
                
                if ~obj.opts.saveMomentum
                    state.momentum = [] ;
                end
            end
        end 
        
        function predictedLabelMap = classifyOn(obj, evalFeatureCube, ...
                maskMap)
            % TODO: Implement
            predictedLabelMap = [];
        end
        
    end
end


function layers = fillMissingInitialValues(layers)
    for i = 1:numel(layers)
        J = numel(layers{i}.weights) ;
        if ~isfield(layers{i}, 'learningRate')
            layers{i}.learningRate = ones(1, J) ;
        end
        if ~isfield(layers{i}, 'weightDecay')
            layers{i}.weightDecay = ones(1, J) ;
        end
    end
end

function [featureCube, mean, std] = normalizeData(featureCube, labelMap)
    % Initialize mean and std
    numFeatureDims = size(featureCube, 3);
    mean = zeros(1, numFeatureDims);
    std = zeros(1, numFeatureDims);
    
    valid = find(labelMap >= 0);
    for dim = 1 : numFeatureDims
        slice = featureCube(:, :, dim);
        
        % save mean and std
        mean(dim) = mean(slice(valid));
        std(dim) = std(slice(valid));
        
        % normalize slice
        featureCube(:, :, dim) = ...
            (featureCube(:, :, dim) - obj.mean(dim)) / obj.std(dim);
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
        case 'gpu', moveop = @(x) gpuArray(x) ;
        case 'cpu', moveop = @(x) gather(x) ;
        otherwise, error('Unknown destination ''%s''.', destination) ;
    end

    if numGPUs >= 1
        % Get logger
        logger = Logger.logger();
        logger.debug('ConvNet', ['Move network to ' destination]);
        
        % Move network to destination
        net = vl_simplenn_move(net, destination);
        
        % Move state to destination
        for i = 1:numel(state.momentum)
            for j = 1:numel(state.momentum{i})
                state.momentum{i}{j} = moveop(state.momentum{i}{j});
            end
        end
    end
end
