classdef ConvNet < Classifier
%CONVNET convolutional neural net
%
%    Convolutional neural network that extracts training batches from a
%    given feature cube and learns on them for a given number of epochs.
%    The trained model can then be used for classification.
%    Make sure that matConvNet has been compiled by running `vl_compilenn`.
%
%% Properties:
%    opts . options regarding training
%    net .. network structure
%    mean . mean of training data for each input dimension
%    std .. standard deviation of training data for each input dimension
%
%% Methods:
%    ConvNet ....... Constructor. Can take Name, Value pair arguments 
%                    that change the internal parameters of the net.
%    toString ...... See documentation in superclass Classifier.
%    toShortString . See documentation in superclass Classifier.
%    trainOn ....... See documentation in superclass Classifier.
%    classifyOn .... See documentation in superclass Classifier.
%
% Version: 2017-02-10
% Author: Marianne Stecklina
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
            str = 'convNet';
        end
        
        function str = toShortString(obj)
            str = 'convNet';
        end
        
        function obj = trainOn(obj, trainFeatureCube, trainLabelMap)
            % Create network structure
            global NUMCLASSES;
            obj.net = createNet(obj.opts.sampleSize, numDim, NUMCLASSES,...
                obj.opts.filterSize);
            
            % Initialize net
            for i=1:numel(obj.net.layers)
                J = numel(obj.net.layers{i}.weights) ;
                if ~isfield(obj.net.layers{i}, 'learningRate')
                    obj.net.layers{i}.learningRate = ones(1, J) ;
                end
                if ~isfield(obj.net.layers{i}, 'weightDecay')
                    obj.net.layers{i}.weightDecay = ones(1, J) ;
                end
            end
            
            % preprocessing: normalize data to mean 0 and variance 1
            valid = find(trainLabelMap >= 0);
            for dim = 1 : size(trainFeatureCube, 3)
                slice = trainFeatureCube(:, :, dim);
                % save mean and std
                obj.mean(dim) = mean(slice(valid));
                obj.std(dim) = std(slice(valid));
                % normalize slice
                trainFeatureCube(:, :, dim) = ...
                    (trainFeatureCube(:, :, dim) ...
                    - obj.mean(dim)) / obj.std(dim);
            end
            clear('valid', 'slice');
            
            state = [];
            % get shuffeled list of labeled pixels
            [labeled(1, :), labeled(2, :)] = find(trainLabelMap > 0);
            labeled = labeled(:, randperm(size(labeled, 2)));
            
            % TODO: load snapshot and start with epoch n
            for epoch = 1 : obj.opts.numEpochs
                % initialize with momentum 0
                if isempty(state) || isempty(state.momentum)
                    for i = 1:numel(obj.net.layers)
                        for j = 1:numel(obj.net.layers{i}.weights)
                            state.momentum{i}{j} = 0;
                        end
                    end
                end

                % move CNN  to GPU as needed
                numGpus = numel(obj.opts.gpus);
                if numGpus >= 1
                    obj.net = vl_simplenn_move(obj.net, 'gpu');
                    for i = 1:numel(state.momentum)
                        for j = 1:numel(state.momentum{i})
                            state.momentum{i}{j} = ...
                                gpuArray(state.momentum{i}{j});
                        end
                    end
                end
                
                res = [] ;
                error = [] ;
                
                numBatches = ceil(length(labeled) / obj.opts.batchSize);
                
                for batchIndex = 1 : numBatches    
                    % create batch from indices
                    batch = createBatch(labeled, batchIndex, ...
                        obj.opts.batchSize, obj.opts.sampleSize, ...
                        trainFeatureCube, trainLabelMap);

                    if numGpus >= 1
                        batch.data = gpuArray(batch.data) ;
                    end

                    obj.net.layers{end}.class = batch.labels;
                    % train net
                    res = vl_simplenn(obj.net, batch.features, 1, res, ...
                        'backPropDepth', obj.opts.backPropDepth, ...
                        'cudnn', obj.opts.cudnn);

                    % accumulate errors
                    error = sum([error, [... 
                        sum(double(gather(res(end).x)));
                        reshape(obj.opts.errorFunction(obj.opts, ...
                            labels, res), [], 1); ]], 2);

                    % accumulate gradients
                    % TODO: change batchSize to real size of batch which
                    % might be smaller
                    [obj.net, res, state] = accumulateGradients(...
                        obj.net, res, state, obj.opts, ...
                        obj.opts.batchSize, []) ;
                end
                
                if ~obj.opts.saveMomentum
                    state.momentum = [] ;
                else
                    for i = 1:numel(state.momentum)
                        for j = 1:numel(state.momentum{i})
                            state.momentum{i}{j} = ...
                                gather(state.momentum{i}{j}) ;
                        end
                    end
                end

                obj.net = vl_simplenn_move(obj.net, 'cpu') ;
            end
        end 
        
        function predictedLabelMap = classifyOn(obj, evalFeatureCube, ...
                maskMap)
            % TODO: implement
            predictedLabelMap = [];
        end
        
    end
end

