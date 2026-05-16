function UVE=mcuvepls(X,Y,A,method,N,ratio)
%+++ uninformative variable elimination(UVE)-PLS.
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A: The max PC for cross-validation
%            N: The number of Monte Carlo Simulation.
%        ratio: The ratio of calibration samples to the total samples.
%       method: pretreatment method. Contains: autoscaling,
%               pareto,minmax,center or none.

%+++ Default parameters
if nargin<6;ratio=0.75;end
if nargin<5;N=1000;end
if nargin<4;method='center';end
if nargin<3;A=2;end

%+++ Monte Carlo Uninformative Variable Elimination.
[Mx,Nx]=size(X);
K=floor(Mx*ratio);
A=min([size(X) A]);
C=zeros(N,Nx);  

for group=1:N
      temp=randperm(Mx);
      calk=temp(1:K);      
      Xcal=X(calk,:);ycal=Y(calk);  
      PLS=pls(Xcal,ycal,A,method);
      coef=PLS.regcoef_pretreat;
      C(group,:)=coef;    
      if mod(group,100)==0;fprintf('The %d/%dth sampling for MC-UVE finished.\n',group,N);end
  end
U=mean(C);  
SD=std(C);  
RI=U./SD;
[RIs,indexRI]=sort(-RI);
%+++ Output
UVE.RI=RI;
UVE.SortedVariable=indexRI;
UVE.regcoef=C;
UVE.MEAN=U;
UVE.SD=SD;

%+++ There is a song you like to sing.

