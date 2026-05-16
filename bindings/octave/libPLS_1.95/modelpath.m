function path=modelpath(X,y,A,method,alpha)
%+++ Compute the model path from PCR to PLS using PCA.
%+++ Central South University, Changsha 410083, P.R. China.
%+++ Advisor: Prof. Yizeng Liang, yizeng_liang@263.net.
%+++ Coder: Hongdong Li, Dec. 25, 2009, lhdcsu@gmail.com.

%+++ Parameter settings
if nargin<5;alpha=linspace(0,1,11);end
if nargin<4;method='center';end
if nargin<3;A=2;end
if min(alpha)<0|max(alpha)>1; alpha=0.5;end

C=zeros(length(alpha),size(X,2));
for i=1:length(alpha)
  F=ecr(X,y,A,method,alpha(i));
  C(i,:)=F.regcoef(:,end)';
  fprintf('The %dth model finished.\n',i);
end


[U,S,V]=svd(C,0);
%+++ Output
path=U(:,1:2);