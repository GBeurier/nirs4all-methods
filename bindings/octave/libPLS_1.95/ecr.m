function F=ecr(X,y,A,method,alpha)
%+++  Elastic Component Regression, a bridge between PCR and PLS. 
%+++  alpha: 0  PCR
%            1  PLS
%            (0, 1) transitional models between PCR and PLS
%+++  Central South University, Changsha 410083, P.R. China.
%+++  Advisor: Yizeng Liang, yizeng_liang@263.net
%+++  Author: Hongdong Li, lhdcsu@gmail.com
%+++  Dec. 25, 2009


%+++ Parameter settings
if nargin<5;alpha=0.5;end
if nargin<4;method='center';end
if nargin<3;A=2;end
if alpha<0 | alpha>1; alpha=0.5;end


[X,xpara1,xpara2]=pretreat(X,method);
[y,ypara1,ypara2]=pretreat(y,method);
Xorig=X;

[Mx,Nx]=size(X);
A=min([Mx-1 Nx-1 A]);

ssqX=sum(sum((X.^2)));  %+++ Total Variance of X
ssqY=sum(y.^2);         %+++ Total Variance of Y

T=zeros(Mx,A);
P=zeros(Nx,A);
W=zeros(Nx,A);
R=zeros(1,A);
B=zeros(Nx,A);
R2X=zeros(1,A);
R2Y=zeros(1,A);


for i=1:A
  H=(1-alpha)*X'*X+alpha*X'*y*y'*X;
  [w,eiv]=powermethod(H);
 
  t=X*w;
  p=X'*t/(t'*t);
  r=y'*t/(t'*t);
  
  W(:,i)=w;
  T(:,i)=t;
  P(:,i)=p;
  R(i)=r;
   
  B(:,i)=W(:,1:i)*(P(:,1:i)'*W(:,1:i))^(-1)*R(1:i)';

  R2X(i)=(T(:,i)'*T(:,i))*(P(:,i)'*P(:,i))/ssqX*100;
  R2Y(i)=(T(:,i)'*T(:,i))*(R(i)'*R(i))/ssqY*100;
  
  
  X=X-t*p';
  y=y-t*r';
end

Wstar=W*(P'*W)^(-1);  %+++ X*Wstar=T.
% B=Wstar*R';

yfit=Xorig*B*ypara2+ypara1;

%+++ Output
F.method=method;
F.alpha=alpha;
F.xpara=[xpara1;xpara2];
F.ypara=[ypara1;ypara2];
F.regcoef=B;
F.yfit=yfit;
F.wstar=Wstar;   %+++ X x wstar= T
F.xweight=W;
F.xscores=T;
F.xloadings=P;
F.yloadings=R;
F.R2X=R2X;
F.R2Y=R2Y;




