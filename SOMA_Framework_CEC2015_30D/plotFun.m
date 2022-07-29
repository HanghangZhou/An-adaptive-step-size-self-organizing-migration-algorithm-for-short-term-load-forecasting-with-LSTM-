function plotFun(var)
% var's type is 'cell'.
   [row, column] = size(var);
   for i = 1 : row
       figure(i);
       [py, px] = size(var{i});
       x = 1:px;
       plot(x,var{i});          
end