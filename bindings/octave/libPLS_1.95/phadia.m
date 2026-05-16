function F=phadia(X,y,A,method,N,Q,K)
%+++ a Model Population Analysis (MPA) based method for computing a phase diagram
%    for variable slection.
%       proposed by Hongdong Li and Prof. Yizeng Liang.
%+++ X: sample matrix: M x P.
%+++ y: Response variable: M-dimensional.
%+++ N: Number of Monte Carlo Sampling, default value:10000.
%+++ Q: The number of variables randomly selected to build a PLS-LDA model, default 10.
%+++ K: The number of groups for cross validation, default 3.
%+++ Tutor: Yizeng Liang, yizeng_liang@263.net.
%+++ Author: Hong-Dong Li,Jul.18, 2009, lhdcsu@gmail.com.

%+++ Default parametrical settings.
if nargin<7;K=3;end;
if nargin<6;Q=10;end;
if nargin<5;N=10000;end;
if nargin<4;method='autoscaling';end;
if nargin<3;A=3;end;
tic;
[Mx,Nx]=size(X);    %  Data size.
A=min([size(X) A]); %  Latent Variable correction.
%+++ Use a txt file to store the sampled variables. 
PredError=zeros(N,1); 
nLV=zeros(N,1);
model{Nx}=[];
ycal=y;
%+++ Main loop for calculate the prediction errors for each sample.
for i=1:N
    perm=randperm(Nx); 
    vsel=perm(1:Q);              %+++ Randomly select variables.
    for jj=1:Q; temp=model{vsel(jj)};model{vsel(jj)}=[temp;i];end
    Xcal=X(:,vsel);          
    
    %+++ cross validation
    CV=plsldacv(Xcal,ycal,A,K,method,0,0,0);    
    PredError(i)=CV.error_min;
    nLV(i)=CV.optLV;
    if rem(i,100)==0;fprintf('The %d/%dth sampling for variable selection finished.\n',i,N);end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('***>>> Evaluating variable importance...\n');
%+++ Variable assessment by computing DMEAN and DSD: two statistics
InformativeVariable=zeros(1,Nx);
p=zeros(1,Nx);
MEAN0=[];MEAN1=[];
SD0=[];SD1=[];
for i=1:Nx
    v=zeros(Nx,1);
    v(model{i})=1;
    k0=find(v==0);
    k1=find(v==1);
    error1=PredError(k1); 
    error0=PredError(k0);
    MEAN0(i)=mean(error0);
    MEAN1(i)=mean(error1);
    SD0(i)=std(error0);
    SD1(i)=std(error1);
      
    %+++ Computing p-value for each variable.
    [pi,h] = ranksum(error0,error1); 
    if pi<10^(-200);pi=10^(-200);end
    p(i)=pi;
%     condition1=sign(mean(error0)-mean(error1));
%     condition2=sign(std(error0)-std(error1));
%     if condition1==1 && condition2 == 1 && pi < 0.05   
%         p(i)=pi;
%         InformativeVariable(i)=1;
%     else
%         p(i)=1+pi;
%     end   
    
end

DMEAN=MEAN0-MEAN1;
DSD=SD0-SD1;

fprintf('***>>> Evaluating finished.\n');

% [sortedp,indexp]=sort(p);
% COSS=-log10(p); %+++ COSS value.

fprintf('***>>> PS: Variable Selection based on Model Population Analysis.\n');
toc;
%+++ Output
F.method=method;
F.parameter_meaning='N Q K';
F.parameters=[N Q K];
F.TimeMinutes=toc/60;
F.model=model;
F.nLV=nLV;
F.PredError=PredError;
F.DMEAN=DMEAN;
F.DSD=DSD;
F.p=p;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  END  %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

