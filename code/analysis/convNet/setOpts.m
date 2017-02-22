function opts = setOpts(varargin)
%SETOPTS set options of the convolutional net
%
%   Arguments passed to the contructor are handled by this functions.
%   Options that were not specified in the constructor call are set to
%   default values.
%
%%  Input:
%       varargin ..... name, value pairs passed to the ConvNet constructor
%
% Version: 2017-02-10
% Author: Marianne Stecklina
%%

    % create input parser
    p = inputParser;
    
    % set default values
    p.addParameter('expDir', fullfile('data','exp'));
    p.addParameter('continue', true);
    p.addParameter('batchSize', 256);
    p.addParameter('numSubBatches', 1);
    p.addParameter('train', []);
    p.addParameter('val', []);
    p.addParameter('gpus', []);
    p.addParameter('prefetch', false);
    p.addParameter('numEpochs', 300);
    p.addParameter('learningRate', 0.001);
    p.addParameter('weightDecay', 0.0005);
    p.addParameter('momentum', 0.9);
    p.addParameter('saveMomentum', true);
    p.addParameter('nesterovUpdate', false);
    p.addParameter('randomSeed', 0);
    p.addParameter('memoryMapFile', fullfile(tempdir, 'matconvnet.bin'));
    p.addParameter('profile', false);
    p.addParameter('filterSize', 5);
    p.addParameter('sampleSize', 21);

    p.addParameter('conserveMemory', true);
    p.addParameter('backPropDepth', +inf);
    p.addParameter('sync', false);
    p.addParameter('cudnn', true);
    p.addParameter('errorFunction', @errorMulticlass);
    p.addParameter('errorLabels', {'top1err', 'top5err'});
    p.addParameter('plotErrorRates', false);
    p.addParameter('plotDiagnostics', false);
    p.addParameter('plotStatistics', false);
    p.addParameter('stoppingEpochWindow', 100);
    p.addParameter('stoppingErrorMargin', 0.001);
    
    % parse input arguments
    p.parse(varargin{:});

    opts = p.Results;
end

