function [X1,Y1,N,X2,Y2] = svm_demo(C1,C2,m,k,varargin)
% SVM_DEMO Shows a figure and asks for data points to enter before
% transductive SVM is being applied on them (start with enter key).
%
%   Training data:
%       Class 1 = 1 key (left mouse click)
%       Class 2 = 2 key (right mouse click)
%       Class 0 = 0 key (middle mouse click)
%       Classes 3 to 9 = 3 to 9 key
%
%   Test data:
%       Classes 1 to 9 = shift + 1 to 9 key
%
% Classes 1 to 9 contain labeled data; Class 0 contains unlabeled data.
%
% Press enter key to perform transductive SVM on the given training data.
% Press space bar to classify a new data point by means of the previously
% learned model. Press escape to terminate the program.
%
% Input:
%       C1(1,1) = Misclassification penalty (labeled data)
%       C2(1,1) = Misclassification penalty (neutral data)
%       m = Multiclass SVM function handle
%       k = Kernel function handle
%
% Optional input (in given order):
%       X1(m1,2) = Training data points
%       Y1(m1,1) = Training data labels
%       N (m2,2) = Training data points (neutral)
%       X2(m3,2) = Test data points
%       Y2(m3,1) = Test data labels
%
% Output:
%       X1(n1,2) = Training data points
%       Y1(n1,1) = Training data labels
%       N (n2,2) = Training data points (neutral)
%       X2(n3,2) = Test data points
%       Y2(n3,1) = Test data labels

% Define markers
marker_trn = 'o';
marker_tst = '^';
msize = 7;

% Define class grid
s = 100; grid = 0:2:s;

% Open demo figure
figure(1); title('SVM DEMO');
axis([0,s,0,s]); hold on;

% Get input data
varsize = size(varargin,2);

if varsize >= 2,
    X1 = varargin{1,1};
    Y1 = varargin{1,2};
else
    X1 = []; % Training data points
    Y1 = []; % Training data labels
end

if varsize >= 3,
    N  = varargin{1,3};
else
    N  = []; % Training data points (neutral)
end

if varsize >= 5,
    X2 = varargin{1,4};
    Y2 = varargin{1,5};
else
    X2 = []; % Test data points
    Y2 = []; % Test data labels
end

% Show data points
show_data();

% Perform demo
XS = -1;

while 1,
    % Fetch data point
    [x,y,b] = ginput(1); point = [x,y];

    % Check data point
    if size(x,1) > 0,
        % Reset marker
        marker = marker_trn;

        % Check button
        switch b
            % Training data (class 0, mouse and key)
            case {2,48}
                set = 0; class = 0;

            % Training data (class 1, mouse)
            case 1
                set = 1; class = 1;

            % Training data (class 2, mouse)
            case 3
                set = 1; class = 2;

            % Training data (class 1 to 9)
            case {49,50,51,52,53,54,55,56,57}
                set = 1; class = b - 48;

            % Test data (class 1 to 9)
            case {33,34,167,36,37,38,47,40,41}
                % Correct b
                if b == 167,
                    b = 35;
                elseif b == 47
                    b = 39;
                end

                set = 2; class = b - 32;
                marker = marker_tst;

            % To be classified data
            case 32
                if XS ~= -1,
                    % Get color
                    color = color_fg(f(point));

                    % Set point
                    plot(x,y,marker_trn,'MarkerEdgeColor',color,'MarkerFaceColor',color,'MarkerSize',5);
                end

                continue;

            % Exit program
            case 27
                close(1); break;

            % Skip iteration
            otherwise
                continue;
        end

        % Add data point...
        switch set
            case 1 % ...to training set
                X1(size(X1,1)+1,1:2) = point ;
                Y1(size(Y1,1)+1,1:1) = class ;

            case 2 % ...to test set
                X2(size(X2,1)+1,1:2) = point ;
                Y2(size(Y2,1)+1,1:1) = class ;

            case 0 % ...to training set
                N(size(N,1)+1,1:2) = point ;
        end

        % Draw data point
        plot(x,y,marker,'MarkerEdgeColor','k','MarkerFaceColor',color_fg(class),'MarkerSize',msize,'LineWidth',2);
    else
        % Compute hyperplane
        [f,XS,~,~,t] = m(X1,Y1,N,C1,C2,k);

        % Reset figure
        clf(1); title(['SVM DEMO (cpu time: ' , num2str(t) , 's)']); axis([0,s,0,s]); hold on;

        % Show class areas
        for y = grid,
            for x = grid,
                % Get color
                color = color_bg(f([x,y]));

                % Draw point
                plot(x,y,marker_trn,'MarkerEdgeColor',color,'MarkerFaceColor',color,'MarkerSize',3);
            end
        end

        % Show data points
        show_data();

        % Mark support vectors
        plot(XS(:,1),XS(:,2),'k+','MarkerSize',msize-2);
    end
end


%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary function %
%%%%%%%%%%%%%%%%%%%%%%

    function show_data()
        % Training data
        for i = 1 : size(X1,1),
            plot(X1(i,1),X1(i,2),marker_trn,'MarkerEdgeColor','k','MarkerFaceColor',color_fg(Y1(i)),'MarkerSize',msize,'LineWidth',2);
        end

        % Training data (neutral)
        for i = 1 : size(N,1),
            plot(N(i,1),N(i,2),marker_trn,'MarkerEdgeColor','k','MarkerFaceColor',color_fg(0),'MarkerSize',msize,'LineWidth',2);
        end

        % Test data
        for i = 1 : size(X2,1),
            plot(X2(i,1),X2(i,2),marker_tst,'MarkerEdgeColor','k','MarkerFaceColor',color_fg(Y2(i)),'MarkerSize',msize,'LineWidth',2);
        end
    end

end