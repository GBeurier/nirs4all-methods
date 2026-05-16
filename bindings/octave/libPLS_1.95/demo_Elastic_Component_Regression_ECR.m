%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%           This is DEMO                                           %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% ECR is a general regression method. It has one supervising       %%%%%%%%%%%%%%%%%%%
%%%%%%% factor alpha=[0,1]                                               %%%%%%%%%%%%%%%%%%%
%%%%%%% when alpha=0,   ECR corresponds to PCR                           %%%%%%%%%%%%%%%%%%%
%%%%%%% when alpha=1,   ECR corresponds to PLS                           %%%%%%%%%%%%%%%%%%%
%%%%%%% when 0<alpha<1, ECR gives transitional models betwen PCR and PCR %%%%%%%%%%%%%%%%%%%
%%%%%%% How to use this script: just run it line by line and check       %%%%%%%%%%%%%%%%%%%
%%%%%%% out the results from each line.                                  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%+++ simulate data
clear;close all;
n=25;p=50;
Nnoise=50;
x=rand(n,p);
x=pretreat(x,'center');
[u,s,v]=svd(x);
d=diag(s);
d=d(1:5)/sum(d);
T=x*v(:,1:5);
X0=T*v(:,1:5)';
noise=randn(n,p+Nnoise);
noise=0.005*noise/max(max(abs(noise)));
X=[X0 rand(n,Nnoise)]+noise; 
y=T*[5 4 3 2 1]';

%+++ two way cross validation of ECR.
A=5;
K=5;
method='center';
CV=ecrcv(X,y,A,K,method)
surf(CV.ALPHA,CV.LV,CV.RMSECV)
xlabel('alpha');ylabel('nLV');
zlabel('RMSECV');
set(gcf,'color','white');

%+++ build an ECR model and make predictions
A=5;
alpha=0.5;  % this is to build a transitional model at alpha=0.5. If alpha=0 or 1, the model will be PCR or PLS, respectively.
Xcal=X(1:15,:);ycal=y(1:15);
Xtest=X(15:end,:);ytest=y(15:end);
ECR=ecr(Xcal,ycal,A,method,alpha);
PRED=ecrpred(ECR,Xtest,ytest);
figure;
plot(ytest,PRED.ypred,'.',ytest,ytest,'r-');
xlabel('Observed');
ylabel('Predicted');
set(gcf,'color','white');

%+++ show path from PCR to PLS
path=modelpath(X,y,A,method);
figure;
plotpath(path);
title('Path from PCR to PLS');
set(gcf,'color','white');


