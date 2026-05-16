function [WP,RMSEF]=mwpls(X,y,A,width)
%+++ moving window PLS
%+++ A: number of maximal latent variables;
%+++ width: window width, must be a odd number, e.g. 5,7,15 etc



if nargin<4;width=15;end
if nargin<3;A=5;end

[Mx,Nx]=size(X);
X=pretreat(X,'center');
y=pretreat(y,'center');
%+++ main
w=(width-1)/2;
WP=w+1:Nx-w;
A=min([A 2*w+1]);
RMSEF=zeros(length(WP),A);R2=[];
for i=w+1:Nx-w     
     Xsub=[X(:,i-w:i+w)];
     B=pls_nipals_in(Xsub,y,A);
     yest=Xsub*B;
     error=(yest-repmat(y,1,A)).^2;
     RMSEF(i-w,:)=sqrt(mean(error));
     fprintf('The %d window finished.\n',i);
end


%++++++++++++
%+++ sub-function
function [B,W,T,P,Q]=pls_nipals_in(X,Y,A)

[n,p]=size(X);
A=min([n p A]);
Xorig=X;
Yorig=Y;

ssqX=sum(sum((X.^2)));
ssqY=sum(Y.^2);
B=zeros(p,A);
for a=1:A    
    W(:,a)=X'*Y;
    W(:,a)=W(:,a)/norm(W(:,a));
    T(:,a)=X*W(:,a);
    P(:,a)=X'*T(:,a)/(T(:,a)'*T(:,a));
    Q(a,1)=Y'*T(:,a)/(T(:,a)'*T(:,a));
    X=X-T(:,a)*P(:,a)';
    Y=Y-T(:,a)*Q(a,1);
    
    B(:,a)=W(:,1:a)*(P(:,1:a)'*W(:,1:a))^(-1)*Q(1:a);

    R2X(a,1)=(T(:,a)'*T(:,a))*(P(:,a)'*P(:,a))/ssqX*100;
    R2Y(a,1)=(T(:,a)'*T(:,a))*(Q(a,1)'*Q(a,1))/ssqY*100;
   
end
