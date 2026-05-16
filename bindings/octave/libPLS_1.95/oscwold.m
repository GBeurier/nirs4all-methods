function F=oscwold(Xcal,Ycal,Xtest,nOSC)
%+++ Orthogonal Signal Correction using Wold's method
%+++ nOSC: number of OSC components, default 1
%+++ Author: Hongdong Li,lhdcsu@gmail.com, Mar. 30, 2012
%+++ reference:Svante Wold et al, Orthogonal signal correction of near-infrared spectra
%    Chemometrics and Intelligent Laboratory Systems, 44(1998)175?185


if nargin<4;nOSC=1;end

[n,p]=size(Xcal);
D=eye(n);
tol=0.000001;
SSQcal=sum(sum(Xcal.^2));
SSQtest=sum(sum(Xtest.^2));

PROJ=(D-Ycal*inv(Ycal'*Ycal)*Ycal');
for i=1:nOSC
    [U,S,V]=svds(Xcal,1);
    t_old=U(:,1)*S(1,1);
    error=1;
    while error>tol
      t=PROJ*t_old;
      [w]=pls_nipals(Xcal,t,min(size(Xcal-1)));
      t=Xcal*w;
      error=norm(t-t_old)/norm(t_old);
      t_old=t;
    end
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



