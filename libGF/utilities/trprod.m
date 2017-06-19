function t = trprod(A,B)
% this function is a more efficient way of computing trace(A*B)

t = sum(sum(A.*(B.')));

end