function F=mcuveplslda(X,Y,A,N,ratio,method,weight,OPT)
%+++ uninformative variable elimination(UVE)-PLS.
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A: The number of latent variables
%            N: The number of Monte Carlo Simulation.
%        ratio: The ratio of calibration samples to the total samples.
%       method: pretreatment method. Contains: autoscaling,
%               pareto,minmax,center
%       weight: 0 or 1, whether to weight samples

%+++ Default parameters
if nargin<8;OPT=1;end
if nargin<7;weight=0;end;
if nargin<6;method='autoscaling';end
if nargin<5;ratio=0.7;end
if nargin<4;N=1000;end
if nargin<3;A=2;end



%+++ Monte Carlo Uninformative Variable Elimination.
[Mx,Nx]=size(X);
Q=floor(Mx*ratio);
A=min([size(X) A]);
C=sparse(N,Nx);  

for group=1:N
      temp=randperm(Mx);
      calk=temp(1:Q);      
      Xcal=X(calk,:);ycal=Y(calk);  
      PLSLDA=plslda(Xcal,ycal,A,method,weight);
      coef=PLSLDA.regcoef_lda_origin;
      C(group,:)=coef(1:end-1)';    
      if OPT==1; fprintf('The %dth sampling for MC-UVE-PLSLDA finished.\n',group);end
  end
  U=mean(C);  SD=std(C);  RI=abs(U./SD);
  [RIs,indexRI]=sort(-RI);
  Vsel=indexRI;
  

%+++ Output
F.RI=RI;
F.SortedVariable=Vsel;
F.Coefficient=C;
%+++ There is a song you like to sing.


