function [ACC] = ACC_Lei(gnd,res)
%ACC_lei: get the ACC score for a clustering result
%   [ACC] = ACC_lei(gnd,res);
%
%
%   

%==========
res = bestMap(gnd,res);
ACC = length(find(gnd == res))/length(gnd);