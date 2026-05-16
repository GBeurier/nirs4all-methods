function F=vcn(X,y,A,method,N,Q,K)
%+++ Variable Complementary Network based on PLS-LDA.
%+++ X: sample matrix: M x P.
%+++ y: Response variable: M-dimensional.
%+++ A: number of latent variables in PLS-LDA.
%+++ N: Number of Monte Carlo Sampling, default value:10000.
%+++ Q: The number of variables randomly selected to build a PLS-LD model, default 3.
%+++ K: The number of groups for cross validation, default 3.
%+++ Tutor: Yizeng Liang, yizeng_liang@263.net.
%+++ Coded by Hong-Dong Li,Jul.18, 2009, lhdcsu@gmail.com.


%+++ Default parametrical settings.
if nargin<7;K=5;end;
if nargin<6;Q=2;end;
if nargin<5;N=10000;end;
if nargin<4;method='autoscaling';end;
if nargin<3;A=3;end;
tic;
[Mx,Nx]=size(X);    %  Data size.
A=min([size(X) A]); %  Latent Variable correction.
%+++ Use a txt file to store the sampled variables.
PredError=nan(N,1); 
coef=nan(N,Nx);
nLV=nan(N,1);
Q2=nan(N,1);
ycal=y;
model{Nx}=[];
%+++ Main loop for calculate the prediction errors for each sample.
i=1;
while i<=N
    perm=randperm(Nx); 
    vsel=perm(1:Q);              %+++ Randomly select variables.
    for jj=1:Q; temp=model{vsel(jj)};model{vsel(jj)}=[temp;i];end
    Xcal=X(:,vsel);
    %+++ cross validation
    CV=plsldacv(Xcal,ycal,A,K,method,0,0);   % revision  
    PLSLDA=plslda(Xcal,ycal,CV.optLV,method);
    coef(i,vsel)=PLSLDA.regcoef_lda_origin(1:end-1)';
    PredError(i)=CV.error_min;    %revision
    nLV(i)=CV.optLV;          %revison
    if rem(i,100)==0;fprintf('The %d/%dth sampling for variable selection finished.\n',i,N);end
    i=i+1;
end

%+++ %%%%%%%%%%%%%%%%%%%++++++++++++++++++
fprintf('***>>> Computing variable complementary information...\n');
VCN=computeCI(model,coef,PredError,N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('***>>> PS: Variable Selection based on Model Population Analysis.\n');
fprintf('***>>> PS: There is a song you like to sing.\n');
toc;
time=[num2str(toc/60) ' minutes'];
%+++ Output
F.method=method;
F.parameter_meaning='N Q K';
F.parameters=[N Q K];
F.VCN=VCN;
F.model=model;
F.coef=coef;
F.nLV=nLV;
F.PredError=PredError;
F.TimeCost=time;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% sub function   %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Z=computeCI(model,coef,PredError,N,q)
%+++ Compute complementary information (CI) 
%+++ percentage of models used for calculating VCN, default: 0.05
%+++ output: Z: the variable complementary information network.
if nargin<5;q=0.05;end



coef=abs(coef);
p=size(coef,2);
variableIndex=1:p;
error005q=quantile(PredError,q);
index005=find(PredError<=error005q);
error005=PredError(index005);

coef005=coef(index005,:);
n005=length(index005);
Model005=nan(n005,p);
Z005=zeros(p,p);
maxErr=max(PredError);
for i=1:p
  temp=model{i};
  Model005(:,i)=ismember(index005,temp);  %+++ 0-1 valued...
end


for i=1:n005
  ki=find(Model005(i,:)==1);
  err=error005(i);
  cb=nchoosek(ki,2);
  maxDiffb=max(coef005(i,:))-min(coef005(i,:));
  for j=1:size(cb,1)
    a=cb(j,1);b=cb(j,2);
    diffb=abs((coef005(i,a)-coef005(i,b))); 
    score=(maxErr/(err+0.001))*cos(diffb*pi/2/maxDiffb)*(coef005(i,a)+coef005(i,b))/2;
    Z005(a,b)=Z005(a,b)+score;
    Z005(b,a)=Z005(b,a)+score;  
  end
end
%+++ output
Z=Z005+eps;
Z=Z/max(max(Z));



