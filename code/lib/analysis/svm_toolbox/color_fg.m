function color = color_fg(class)
% COLOR_FG Returns a color vector in RGB format for the given class.
%
% Input:
%       class(1,1) = Class label (from 0 to 9, all others will be white)
%
% Output:
%       color(1,3) = Class color in RGB format [r,g,b] with 0 <= r,g,b <= 1

switch class
    case 1
        color = [1.0,0.0,0.0];

    case 2
        color = [0.0,0.0,1.0];

    case 3
        color = [0.0,1.0,0.0];

    case 4
        color = [1.0,1.0,0.0];

    case 5
        color = [0.0,1.0,1.0];

    case 6
        color = [1.0,0.0,1.0];

    case 7
        color = [0.2,0.5,1.0];

    case 8
        color = [0.9,0.5,0.0];

    case 9
        color = [0.7,0.0,0.9];

    case 0
        color = [0.5,0.5,0.5];

    otherwise
        color = [1.0,1.0,1.0];
end
end