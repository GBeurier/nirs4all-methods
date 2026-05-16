function F=ecrpred(model,Xtest,ytest)
%+++  Make predictions using Elastic Component Regression. 
%+++  Advisor: Yizeng Liang, yizeng_liang@263.net
%+++  Coded by Hongdong Li, lhdcsu@gmail.com
%+++  Central South University, Changsha 410083, P.R. China.
%+++  Dec. 25, 2009


Xtest=pretreat(Xtest,model.method,model.xpara(1,:),model.xpara(2,:));
B=model.regcoef(:,end);
ypred=Xtest*B*model.ypara(2)+model.ypara(1);
error=ypred-ytest;
RMSEP=sqrt(sum(error.^2)/length(ytest));

%+++ Output
F.ypred=ypred;
F.error=error;
F.RMSEP=RMSEP;














