function exportFigure(h, exportFolder, figureName)
%   print(h,'-dpng',[exportFolder figureName])
%   figure(h);
%   set(gca, 'LooseInset', get(gca,'TightInset'));
%   print(h,'-dpdf',[exportFolder figureName])
  set(h, 'color', 'w');
  export_fig(h, [exportFolder '/' figureName '.pdf']);
  saveas(h, [exportFolder '/' figureName '.fig']);
end