
function yd=int_dec(x,I,D,mode)
%
%   YD=INT_DEC(X,I,D,MODE)
%
% Interpolation by a factor I followed by a decimation by a factor D
% 
%  mode:  (optional)
%     'matlab' : implementation with the MATLAB functions interp() and
%                decimate()  [default option]
%     'fft'    : implementation with FFT (I think sometimes can be more
%                efficient). Note: this is for complex X.
%

if nargin<4
	mode='matlab';
end

switch mode
	case 'matlab'
		yd=decimate(interp(x,I),D);		
	case 'fft'	
		warning('have a look at the code, not completely debugged');
		% mean correction: avoid effects of leakage (if you have to decimate a
		% signal with a higher bandwidth than the allowed by I and D, the first
		% sample of yd can be quite different from that of x because of the change
		% of mean that the DFT filtering produces).
		mean_correction=1;
		
		% % %
		N=size(x,2);
		Pcomp=band_indices_comp(N*I,pi/max([I,D]),0);
		y_now=[x; zeros(I-1,N)];
		Y_now=fft(y_now(:));
		mb=mean(Y_now(:));
		Y_now(Pcomp)=0;
		
		if mean_correction
			yp_now=ifft(I*Y_now-I*mean(Y_now(:))+mb);
		else
			yp_now=ifft(I*Y_now);
		end
		yd=yp_now(1:D:end).';
	otherwise 
		error('mode does not exist');
end
return



function S=band_indices(N,B,Omegac)
% the n-th column in S has the indices of the n-th band defined by 

if size(Omegac,1)~=1
	error('Omegac must be a row vector');
end

nband=ceil(N*B/(2*pi));
nband=2*ceil((nband-1)/2)+1; % round to the next odd number
snband=(nband-1)/2;
Omegac_ind=round((N-1)*Omegac/2/pi)+1;


n=-snband:snband;
S=Omegac_ind'*ones(1,nband)+ones(length(Omegac),1)*n;

S=mod(S-1,N)+1;

return

function R=band_indices_comp(N,B,Omegac)
% it is like band_indices(N,B,Omegac), but it returns the indices that do
% not belong to the band (the complementary set).

S=band_indices(N,B,Omegac);
nrows=size(S,1);
nband=size(S,2);
R=zeros(nrows,N-nband);
for n=1:nrows
	R(n,:)=setxor(1:N,S(n,:));
end

return

