
function [N,X]=histogram( d , nbins )
% histograms of the rows of H
nr=size(d,1);
nc=size(d,2);
N=zeros(nr,nbins);
X=zeros(nr,nbins);

for k=1:nr
	[n,x]=hist(d(k,:),nbins);
	N(k,:)=n;
	X(k,:)=x;
end
return
