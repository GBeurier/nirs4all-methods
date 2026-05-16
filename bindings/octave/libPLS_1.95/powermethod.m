function [evec,eval]=powermethod(X)
%+++ Power method for computing inverse of a symmetric matrix.

epslon=1e-10;
error=1;
x0=X(:,1);
while error>epslon
    x1=X*x0;
    lambda=norm(x1);
    x1=x1/lambda;
    error=norm(x1-x0)/norm(x0);
    x0=x1;
end
evec=x1;
eval=lambda;


    

