function F=oscfearn(Xcal,Ycal,Xtest,nOSC)
%+++ Orthogonal Signal Correction using Fearn's method
%+++ nOSC: number of OSC components, default 1
%+++ Author: Hongdong Li,lhdcsu@gmail.com, Mar. 30, 2012
%+++ reference:Tom Fearn, On orthogonal signal correction, 
%    Chemometrics and Intelligent Laboratory Systems 50,2000. 47-52


if nargin<4;nOSC=1;end


[n,p]=size(Xcal);
D=eye(p);
SSQcal=sum(sum(Xcal.^2));
SSQtest=sum(sum(Xtest.^2));

for i=1:nOSC
    M=D-Xcal'*Ycal*inv(Ycal'*Xcal*Xcal'*Ycal)*Ycal'*Xcal;
    [ev,lambda]=eigs(Xcal*M*M'*Xcal',1);
    w=1/sqrt(lambda)*M*Xcal'*ev;
    t=Xcal*w;
    p=Xcal'*t/(t'*t);
    Xcal=Xcal-t*p';
    W(:,i)=w;
    T(:,i)=t;
    P(:,i)=p;   
end
Zcal=Xcal;

%+++ Correcting new samples
for i=1:nOSC
 t=Xtest*W(:,i);
 Xtest=Xtest-t*P(:,i)';
end
Ztest=Xtest;

%+++ ratio of explained variance of OSC components
R2Xcal=1-sum(sum(Zcal.^2))/SSQcal;
R2Xtest=1-sum(sum(Ztest.^2))/SSQtest;

%+++ output
F.W=W;
F.T=T;
F.P=P;
F.Zcal=Zcal;
F.Ztest=Ztest;
F.R2Xcal=R2Xcal;
F.R2Xtest=R2Xtest;



