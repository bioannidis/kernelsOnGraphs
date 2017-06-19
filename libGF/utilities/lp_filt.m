function [Hd,h] = lp_filt( bw , N)
%
% input
%   bw is the bassband bandwidth (i.e., the maximum frequency) in radians
%   N is the number of samples
%  
% output
%   h is a vector with the coefficients of a lowpass FIR filter
%   Hd is a filter object. Use y = filter (Hd, x) to filter
%
d = fdesign.lowpass('N,Fc',N,bw/pi);
Hd=design(d,'window');

h=Hd.Numerator;

return