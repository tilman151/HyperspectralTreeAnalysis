classdef SpaReSSEnse < Classifier
    %SPARESSENS Spatially regularized semi-supervised ensemble
    %   
    %    Based on the scientific work presented in [...]. The algorithm
    %    works as follows:
    %    1. Unsupervised clustering is performed in the spectral domain.
    %    2. For each labeled training sample, the label is propagated to
    %    unlabeled pixels in the spatial neighborhood that belong to the
    %    same cluster.
    %    3. A classifier is trained on this enriched data set.
    %    4. The output image is regularized by assuming spatial smoothness.
    %
    %    Any classifier or ensemble of classifiers following the common 
    %    interface can be used internally. Furthermore, the label 
    %    propagation and the regularization step can optionally be 
    %    disabled.
    %
    %% Properties:
    %    classifier ......... Instance of a Classifier (supervised or 
    %                         semi-supervised, single or ensemble) to be
    %                         used internally.
    %    r .................. Float value. Radius defining the neighborhood
    %                         for label propagation.
    %    doLabelPropagation . Boolean. Enables/disables the label
    %                         propagation step.
    %    doRegularization ... Boolean. Enables/disables the regularization
    %                         step.
    %
    %% Methods:
    %    SpaReSSEnse .. Constructor. Can take up to four input arguments:
    %        classifier ......... Set the classifier property.
    %        r .................. [Optional] Set the radius property.
    %        doLabelPropagation . [Optional] Set the label propagation
    %                             property.
    %        doRegularization ... [Optional] Set the regularization
    %                             property.
    %    trainOn ...... See documentation in superclass Classifier.
    %    classifyOn ... See documentation in superclass Classifier.
    %
    % Version: 2016-12-10
    % Author: Cornelius Styp von Rekowski
    %
    
    properties
        % The internally used superised classifier (ensemble)
        classifier;
        
        % Neighborhood radius
        r = 5;
        
        % Enabled/Disabled processing steps
        doLabelPropagation = true;
        doRegularization = true;
    end
    
    methods
        function obj = SpaReSSEnse(classifier, r, ...
                doLabelPropagation, doRegularization)
            
            % Classifier to be used internally
            obj.classifier = classifier;
            
            % Radius given?
            if nargin > 1
                obj.r = r;
            end
            
            % Processing steps enabled/disabled?
            if nargin > 3
                obj.doLabelPropagation = doLabelPropagation;
                obj.doRegularization = doRegularization;
            end
        end
        
        function obj = trainOn(obj, trainFeatures, trainLabels)
            if obj.doLabelPropagation
                % Perform unsupervised clustering
                clusters = clustering(trainFeatures);
                
                % Propagate labels in spatial neighborhood for matching
                % clusters
                trainLabels = propagateLabels(trainLabels, clusters);
            end
            
            % Train classifier on (enriched) data set
            obj.classifier.trainOn(trainFeatures, trainLabels);
        end
        
        function labels = classifyOn(obj, evalFeatures)
            % Predict labels
            labels = obj.classifier.classifyOn(evalFeatures);
            
            if obj.doRegularization
                % Regularize output labels based on spatial smoothness
                [x, y, ~] = size(evalFeatures);
                labels = regularize(labels, x, y);
            end
        end
    end
    
end


function clusters = clustering(features)
    % TODO: Perform k-means custering
end

function enrichedLabels = propagateLabels(labels, clusters)
    % TODO: Propagate labels
end

function regularizedLabels = regularize(labels, x, y)
    % TODO: Reorder labels to spatial representation
    % TODO: Regularize in spatial domain
end
