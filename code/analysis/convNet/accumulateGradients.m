function [net, res, state] = ...
    accumulateGradients(net, res, state, params, batchSize, parserv)
%ACCUMULATEGRADIENTS Summary of this function goes here
%   Detailed explanation goes here

    numGpus = numel(params.gpus);
    otherGpus = setdiff(1:numGpus, labindex);

    for l=numel(net.layers):-1:1
        for j=numel(res(l).dzdw):-1:1
            
            if ~isempty(parserv)
                tag = sprintf('l%d_%d',l,j);
                parDer = parserv.pull(tag);
            else
                parDer = res(l).dzdw{j} ;
            end

            if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                % special case for learning bnorm moments
                thisLR = net.layers{l}.learningRate(j);
                net.layers{l}.weights{j} = vl_taccum(...
                    1 - thisLR, ...
                    net.layers{l}.weights{j}, ...
                    thisLR / batchSize, ...
                    parDer);
            else
                % Standard gradient training.
                thisDecay = ...
                    params.weightDecay * net.layers{l}.weightDecay(j);
                thisLR = ...
                    params.learningRate * net.layers{l}.learningRate(j);

                if thisLR>0 || thisDecay>0
                    % Normalize gradient and incorporate weight decay.
                    parDer = vl_taccum(1/batchSize, parDer, ...
                        thisDecay, net.layers{l}.weights{j});

                    % Update momentum.
                    state.momentum{l}{j} = vl_taccum(...
                        params.momentum, state.momentum{l}{j}, -1, parDer);

                    % Nesterov update (aka one step ahead).
                    if params.nesterovUpdate
                        delta = vl_taccum(...
                            params.momentum, state.momentum{l}{j}, ...
                            -1, parDer);
                    else
                        delta = state.momentum{l}{j};
                    end

                    % Update parameters.
                    net.layers{l}.weights{j} = vl_taccum(...
                        1, net.layers{l}.weights{j}, thisLR, delta);
                end
            end

            % if requested, collect some useful stats for debugging
            if params.plotDiagnostics
                variation = [];
                label = '';
                switch net.layers{l}.type
                    case {'conv','convt'}
                        variation = ...
                            thisLR * mean(abs(state.momentum{l}{j}(:)));
                        power = mean(res(l+1).x(:).^2);
                        if j == 1 % fiters
                            base = mean(net.layers{l}.weights{j}(:).^2);
                            label = 'filters';
                        else % biases
                            base = sqrt(power);%mean(abs(res(l+1).x(:)));
                            label = 'biases';
                        end
                        variation = variation / base;
                        label = ...
                            sprintf('%s_%s', net.layers{l}.name, label);
                end
                res(l).stats.variation(j) = variation;
                res(l).stats.power = power;
                res(l).stats.powerLabel = net.layers{l}.name;
                res(l).stats.label{j} = label;
            end
        end
    end
end

