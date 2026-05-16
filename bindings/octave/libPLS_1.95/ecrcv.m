function CV=ecrcv(X,y,A,K,method,alpha,OPT,order)
%+++ Two way K-fold Cross-validation for ECR.
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A: The maximal number of latent variables for cross-validation
%            K: fold. when K=m, it is leave-one-out CV
%       method: pretreatment method. Contains: center or autoscaling.
%        alpha: if = 0, ECR corresponds to PCR
%               if = 1, ECR corresponds to PLS
%          OPT: =1 : print process.
%               =0 : don't print process.
%        order: =0 : samples are ordered according to y-values.
%               =1 : samples are randomly partioned for CV.
%+++ Output: Structural data: CV
%+++ Advisor: Prof. Yizeng Liang, yizeng_liang@263.net.
%+++ Coder: Hongdong Li, Dec. 25, 2009, lhdcsu@gmail.com.


if nargin<8;order=0;end
if nargin<7;OPT=1;end;
if nargin<6;alpha=linspace(0,1,11);end
if nargin<5;method='center';end;
if nargin<4;K=10;end;
if nargin<3;A=3;end;


if order==0
  [y,index]=sort(y);
  X=X(index,:);
elseif order==1
  randorder=randperm(length(y));
  X=X(randorder,:);
  y=y(randorder);
end



A=min([size(X) A]);
[Mx,Nx]=size(X);
groups = 1+rem(0:Mx-1,K);
yytest=[];
RMSECV=zeros(A,length(alpha));
Q2=zeros(A,length(alpha));

E=[];           %+++ store predicted values.

for group=1:K
    testk = find(groups==group);  calk = find(groups~=group);
    Xcal=X(calk,:);ycal=y(calk);
    Xtest=X(testk,:);ytest=y(testk);
    
    %+++ pretreatment
    [Xcal,para1,para2]=pretreat(Xcal,method);
    [ycal,ypara1,ypara2]=pretreat(ycal,method);
    Xtest=pretreat(Xtest,method,para1,para2);
    
    TEMP=[];
    for i=1:length(alpha)
       model=ecr(Xcal,ycal,A,'center',alpha(i)); 
       ypred=Xtest*model.regcoef*ypara2+ypara1;
       TEMP=[TEMP ypred];      
    end
    E=[E;TEMP];  
    yytest=[yytest;ytest];
    
    if OPT==1; fprintf('The %dth group finished.\n',group); end;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
%+++ statistics for CV
for i=1:length(alpha)
  temp=E(:,((i-1)*A+1):(i*A)); 
  error=temp-repmat(yytest,1,A);
  PRESS=sum(error.^2);
  cvtemp=sqrt(PRESS/Mx)';
  SST=sumsqr(yytest-mean(yytest));
  for j=1:A
    SSE=sumsqr(temp(:,j)-yytest);
    Q2temp(j,1)=1-SSE/SST;
  end
 
  RMSECV(:,i)=cvtemp;
  Q2(:,i)=Q2temp;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ecrRMSECV=min(min(RMSECV));
ecrQ2=max(max(Q2));
[ecrLV,b]=find(ecrRMSECV==RMSECV);
optAlpha=alpha(b);
[pcrRMSECV,pcrLV]=min(RMSECV(:,1));
[plsRMSECV,plsLV]=min(RMSECV(:,end));
[pcrQ2]=max(Q2(:,1));
[plsQ2]=max(Q2(:,end));

if ecrRMSECV==pcrRMSECV;winner='PCR';elseif ecrRMSECV==plsRMSECV;winner='PLS';else;winner='Transitional model between PCR and PLS';end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LV,ALPHA]=meshgrid(1:A,alpha);
LV=LV';
ALPHA=ALPHA';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ output  %%%%%%%%%%%%%%%%
CV.method=method;
CV.LV=LV;
CV.ALPHA=ALPHA;
CV.ypred=E;
CV.RMSECV=RMSECV;
CV.Q2=Q2;
CV.note1='*****below are comparisons';
CV.description='[PCR  PLS  ECR]';
CV.optLV=[pcrLV plsLV ecrLV];
CV.optRMSECV=[pcrRMSECV plsRMSECV ecrRMSECV];
CV.optQ2=[pcrQ2 plsQ2 ecrQ2];
CV.note2='*****below are best model';
CV.winner=winner;                   % 1: PCR  2:PLS  3:Transitional models in ECR.
CV.minRMSECV=ecrRMSECV;
CV.maxQ2=ecrQ2;
CV.optAlpha=optAlpha;
%+++ There is a song you like to sing.