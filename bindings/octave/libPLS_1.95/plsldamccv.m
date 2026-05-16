function CV=plsldamccv(X,y,A,N,ratio,method,weight,OPT)
%+++ Monte Carlo Cross Validation for PLS-LDA
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A: The max PC for cross-validation
%            N: The number of Monte Carlo Simulation.
%        ratio: The ratio of calibration samples to the total samples.
%       method: pretreatment method. Contains: autoscaling,
%               pareto,minmax,center or none.
%       weight: 0 or 1, whether to weight samples
%          OPT: =1 : print process.
%               =0 : don't print process.
%+++ Output: Structural data: CV
%+++ Hongdong Li, Oct. 16, 2008.
%+++ Revised in Jan.12, 2009.

if nargin<8;OPT=1;end;
if nargin<7;weight=0;end
if nargin<6;method='autoscaling';end
if nargin<5;ratio=0.8;end
if nargin<4;N=1000;end
if nargin<3;A=2;end



A=min([size(X,2) A]);
[Mx,Nx]=size(X);
Q=floor(Mx*ratio);
yytest=[];YR=[];
for i=1:N
    np=randperm(Mx);
    calk=np(1:Q);testk=np(Q+1:end);
    Xcal=X(calk,:);ycal=y(calk);
    Xtest=X(testk,:);ytest=y(testk);
    
    %data pretreatment    
    [Xcal,para1,para2]=pretreat(Xcal,method);
    ycals=ycal;
    
    
    Xtest=pretreat(Xtest,method,para1,para2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [B,wstar,T,P]=plsnipals(Xcal,ycals,A);
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    yr=zeros(Mx-Q,A);
    for j=1:A
        %%%+++ train model.        
        TT=T(:,1:j);
        C=ldapinv(TT,ycal,weight);
        coef=[wstar(:,1:j)*C(1:end-1);C(end)];        
        %+++ predict
        y_est=Xtest*coef(1:end-1)+coef(end);
        yr(:,j)=y_est;
    end
    YR=[YR;yr];yytest=[yytest;ytest];
    if OPT==1;fprintf('The %d/%dth sampling for MCCV PLS-LDA finished.\n',i,N);end;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:A
  error(i)=sum(sign(YR(:,i))~=yytest)/length(yytest);    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[error_min,index]=min(error);

%+++ output  %%%%%%%%%%%%%%%%
CV.method=method;
CV.error=error;
CV.error_min=error_min;
CV.optLV=index;
