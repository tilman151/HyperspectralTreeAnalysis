function color = color_bg(class)
% COLOR_BG Returns a color vector in RGB format for the given class.
%
% Input:
%       class(1,1) = Class label (from 0 to 9, all others will be white)
%
% Output:
%       color(1,3) = Class color in RGB format [r,g,b] with 0 <= r,g,b <= 1

% Get foreground
color = color_fg(class);

% Set background
for i = 1 : 3,
    color(i) = min([1.0,color(i)+0.6]);
end
end