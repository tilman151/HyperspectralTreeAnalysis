function processMats(path)
    %PROCESSMATS Transform .mat files of data in path
    %
    %   The function takes a path to a path to a data directory. Every .png
    %   and .mat file in this directory is transposed, so that the first
    %   dimension is the largest. After that the maximal first dimension of
    %   all files is determined. The cubes and label matrices are padded in
    %   the first dimension to match the maximal one. All files are saved
    %   to .mat files with the suffix _new. Former .png files receive the
    %   suffix _labels.
    %
    %%  Input:
    %       path ... a path to the data directory. Missing last slash is
    %                added
    %
    % Version: 2016-12-08
    % Author: Tilman Krokotsch
    %%
    
    if (path(end) ~= '/' && isunix)
        path = [path, '/'];
    elseif (path(end) ~= '\' && ispc)
        path = [path, '\'];
    end

    files = dir([path, '*.mat']);
    
    if (size(files, 1) == 0)
        disp('No mat files in path. Wrong path?');
        return;
    end
    
    [transpose, maxSizes] = processImages(path);
    
    for i = 1:size(files, 1)
        image = load([path, files(i).name]);
        image = image.cube;

        if transpose(i)
            image = permute(image, [2, 1, 3]);
        end
    
        tic();
        imgSize = size(image);
        cube = zeros(maxSizes(1), size(image, 2), size(image, 3));
        cube(1:imgSize(1), 1:imgSize(2), :) = image;
        name = [path, files(i).name(1:end-4), '_new.mat'];
        save(name, 'cube');
        disp(['Wrote ', files(i).name, ' to ', name, ' in ', num2str(toc), ' seconds.']);
    end
   
end

function [transpose, maxSizes] = processImages(path)
    %PROCESSIMAGES Transform .png files of data in path
    %
    %   The function takes a path to a path to a data directory. Every .png
    %   file in this directory is transposed, so that the first
    %   dimension is the largest. After that the maximal first dimension of
    %   all files is determined. The label matrices are padded in
    %   the first dimension to match the maximal one. All files are saved
    %   to .mat files with the suffix _labels_new.
    %
    %%  Input:
    %       path ... a path to the data directory. Missing last slash is
    %                added
    %
    % Version: 2016-12-08
    % Author: Tilman Krokotsch
    %%
    files = dir([path, '*.png']);
    
    if (size(files, 1) == 0)
        disp('No png files in path. Wrong path?');
        return;
    end
    
    disp(['Process ', num2str(size(files, 1)), ' png files.']);

    images = cell(size(files,1),1);
    for i = 1:size(files, 1)
        images{i} = imread([path, files(i).name]);
    end
    
    sizes = cellfun(@size, images, 'UniformOutput', false);
    sizes = cell2mat(sizes);
    sortedSizes = [max(sizes'); min(sizes')]';
    transpose = sortedSizes(:, 1) == sizes(:, 2);
    
    for i = 1:size(images,1)
        if transpose(i)
            images{i} = images{i}';
        end
    end
    
    maxSizes = max(sortedSizes);
    
    for i = 1:size(images, 1)
        imgSize = size(images{i});
        labels = zeros(maxSizes(1), size(images{i}, 2))-1;
        labels(1:imgSize(1), 1:imgSize(2)) = images{i};
        name = [path, files(i).name(1:end-4), '_labels_new.mat'];
        save(name, 'labels');
    end
    
    disp('Done.');
    
end