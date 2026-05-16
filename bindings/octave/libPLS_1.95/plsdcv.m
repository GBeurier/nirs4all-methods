function DCV=plsdcv(X,y,A,K,method,OPT,order)
%+++ K-fold double cross validation Cross-validation for PLS regression.
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A: The max PC for cross-validation
%            K: fold. when K = m, it is leave-one-out CV
%       method: pretreatment method. Contains: autoscaling, center etc.
%          OPT: =1 Print process.
%               =0 No print.
%               pareto,minmax,center or none.
%+++ Order: =1  sorted, default. For CV partition.
%           =0  random. 
%+++ Output: Structural data: CV
%+++ Hongdong Li, Oct. 16, 2008.
%+++ Revised on Dec.3, 2009.

if nargin<7,order=1;end;
if nargin<6;OPT=1;end;
if nargin<5;method='center';end;
if nargin<4;K=10;end;
if nargin<3;A=2;end;


if order==1
  [y,indexyy]=sort(y);
  X=X(indexyy,:);
else
  indexyy=randperm(length(y));
  X=X(indexyy,:);
  y=y(indexyy);
end


A=min([size(X,1)-ceil(length(y)/K) size(X,2) A]);


yytest=[];
YR=[];
[Mx,Nx]=size(X);
groups = 1+rem(0:Mx-1,K);
yytest=[];yp=[];
nLV=zeros(K,1);
RMSEP=zeros(K,1);
predError=[];

for group=1:K
    
    %+++ generate sub-training set and sub-test set
    calk = find(groups~=group);
    testk = find(groups==group);  
    
    Xcal=X(calk,:);ycal=y(calk);
    Xtest=X(testk,:);ytest=y(testk);
    
    CV=plscv(Xcal,ycal,A,K,method,0,order);
        
    PLS=pls(Xcal,ycal,CV.optLV,method);
    ypre=plsval(PLS,Xtest,ytest);
    residue=ypre-ytest;
    RMSEP(group)=sqrt(sum(residue.^2)/length(ypre));
    
    yytest=[yytest;ytest];
    predError=[predError;residue];
    yp=[yp;ypre;];
    nLV(group)=CV.optLV;       
    if OPT==1;fprintf('The %dth outer loop finished.\n',group);end;
end

RMSECV=sqrt(sum((yp-yytest).^2)/length(yp));
%+++ output
  DCV.method=method;
  DCV.RMSECV=RMSECV;
  DCV.nLV=nLV;
  DCV.RMSEP=RMSEP;
  DCV.predError=predError;

  