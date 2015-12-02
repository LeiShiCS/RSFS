function [Y] = eigY(M,nClass)



[row,col] = size(M);
if row ~= col
	error('M must square matrix!!');
end

[Y, eigvalue] = eig(full(M));
eigvalue = diag(eigvalue);

[junk,index] = sort(eigvalue);

Y = Y(:,index);

if nClass<length(eigvalue)
	Y = Y(:,1:nClass);
end