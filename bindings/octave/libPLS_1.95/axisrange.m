function [low,high]=axisrange(x)
%+++ for a better axis range

r=0.10;
a=min(x);
b=max(x);
d=(b-a);
low=a-d*r;
high=b+d*r;

