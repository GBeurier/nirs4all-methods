function [xbin,ybin]= databin(data,nbins)
%   HISTFIT(DATA,NBINS) plots a histogram of the values in the vector DATA.
%   using NBINS bars in the histogram. With one input argument, NBINS is set 
%   to the square root of the number of elements in DATA. 

%   Copyright 1993-2004 The MathWorks, Inc. 
%   $Revision: 2.13.2.1 $  $Date: 2003/11/01 04:26:28 $
%
if nargin < 2
  nbins = ceil(sqrt(length(data)));
end
distance=(max(data)-min(data));
band=distance/nbins;
[n,xbin]=hist(data,nbins);
ybin=n/band/sum(n);






