function F=opls(X,Y,Xtest,nOSC)
%+++ Function: orthogonal to latent sturctures(O-PLS)
%+++ Input:  X: m x n  (Sample matrix)
%            Y: m x 1  (measured property)
%+++     Xtest: test samples
%         nOSC: The number of OSC components to remove
%+++ Reference: Johan Trygg and Svante Wold, Orthogonal projections to latent
%    structures (O-PLS),J. Chemometrics 2002; 16: 119-128
%+++ Author: by H.D. Li, Fool's day, 2012.
%+++ Contact:lhdcsu@gmail.com

if nargin<4;nOSC=1;end


[n,p]=size(X);
Xorig=X;
ssqXcal=sum(sum((X.^2))); 
ssqXtest=sum(sum((Xtest.^2))); 
w=X'*Y;         % calculate weight vector
w=w/norm(w);    % normalization
for i=1:nOSC 
    t=X*w;                     % calculate scores vector
    p=X'*t/(t'*t);             % calculate loadings of X
    wosc=p-(w'*p)/(w'*w)*w;    % orthogonal weight
    wosc=wosc/norm(wosc);      % normalization
    tosc=X*wosc;               % orthogonal components
    posc=X'*tosc/(tosc'*tosc); % loadings 
    
    X=X-tosc*posc';            % remove orthogonal components
    W_orth(:,i)=wosc;          % record results
    T_orth(:,i)=tosc;          % record results
    P_orth(:,i)=posc;          % record results
    
end


Zcal=X;   %+++ data with orthogonal signal components removed

%+++ Correcting new samples
for i=1:nOSC
 t=Xtest*W_orth(:,i);
 Xtest=Xtest-t*P_orth(:,i)';
end
Ztest=Xtest;

%+++ ratio of explained variance of OSC components
R2Xcal=1-sum(sum(Zcal.^2))/ssqXcal;   
R2Xtest=1-sum(sum(Ztest.^2))/ssqXtest;


%+++ Output
F.W=W_orth;
F.T=T_orth;
F.P=P_orth;
F.Zcal=Zcal;
F.Ztest=Ztest;
F.R2Xcal=R2Xcal;
F.R2Xtest=R2Xtest;





