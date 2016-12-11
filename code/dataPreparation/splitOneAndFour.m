function splitOneAndFour(path)
    %SPLITONEANDFOUR Splits the labels and features of F2G9_01 and F2G9_04
    %
    %   ONLY TO BE USED AFTER EXECUTION OF PROCESSMATS
    %
    %   The function takes a path to the data and splits the labels and
    %   features of F2G9_01 and F2G9_04 into 4 new files:
    %
    %       F2G9_01 --> first half of 01
    %       F2G9_04 --> first half of 01
    %       F2G9_25 --> second half of 01
    %       F2G9_26 --> second half of 04
    %
    %   The files are split in half along the second dimension and get a
    %   padding of one column at the cut, containing -1 as a label.
    %
    %%  Input:
    %       path ... a path to the data directory. Missing last slash is
    %                added
    %
    % Version: 2016-12-09
    % Author: Tilman Krokotsch
    %%
    
    if (path(end) ~= '/' && isunix)
        path = [path, '/'];
    elseif (path(end) ~= '\' && ispc)
        path = [path, '\'];
    end

    warning('off', 'MATLAB:colon:nonIntegerIndex');

    %% Split labels of 01
    load([path, 'F2G9_01_labels_new.mat']);
    labels1 = labels(:,1:size(labels,2)/2);
    labels2 = labels(:,size(labels,2)/2:end);
    
    labels1(:, end+1) = -1;
    labels2 = cat(2, zeros(size(labels2, 1), 1)-1, labels2);
    
    labels = labels1;
    save([path, 'F2G9_01_labels_new.mat'], 'labels');
    labels = labels2;
    save([path, 'F2G9_25_labels_new.mat'], 'labels');
    
    %% Split labels of 04
    load([path, 'F2G9_04_labels_new.mat']);
    labels1 = labels(:,1:size(labels,2)/2);
    labels2 = labels(:,size(labels,2)/2:end);
    
    labels1(:, end+1) = -1;
    labels2 = cat(2, zeros(size(labels2, 1), 1)-1, labels2);
    
    labels = labels1;
    save([path, 'F2G9_04_labels_new.mat'], 'labels');
    labels = labels2;
    save([path, 'F2G9_26_labels_new.mat'], 'labels');
    
    %% Split features of 01
    load([path, 'F2G9_01_new.mat']);
    cube1 = cube(:,1:size(cube,2)/2, :);
    cube2 = cube(:,size(cube,2)/2:end, :);
    
    cube1(:, end+1, :) = 0;
    cube2 = cat(2, zeros(size(cube2, 1), 1, size(cube2, 3))-1, cube2);
    
    cube = cube1;
    save([path, 'F2G9_01_new.mat'], 'cube');
    cube = cube2;
    save([path, 'F2G9_25_new.mat'], 'cube');
    
    %% Split features of 04
    load([path, 'F2G9_04_new.mat']);
    cube1 = cube(:,1:size(cube,2)/2, :);
    cube2 = cube(:,size(cube,2)/2:end, :);
    
    cube1(1, end+1, 1) = 0;
    cube2 = cat(2, zeros(size(cube2, 1), 1, size(cube2, 3))-1, cube2);
    
    cube = cube1;
    save([path, 'F2G9_04_new.mat'], 'cube');
    cube = cube2;
    save([path, 'F2G9_26_new.mat'], 'cube');

    warning('on', 'MATLAB:colon:nonIntegerIndex');

end

