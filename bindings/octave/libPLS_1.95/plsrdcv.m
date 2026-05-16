function RDCV=plsrdcv(X,y,A,K,method,Nmcs,OPT)
%+++ Repeaded double cross validation Cross-validation for PLS regression
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A: The maximal number of latent variables for cross-validation
%            K: fold. when K = m, it is leave-one-out CV
%       method: pretreatment method. Contains: autoscaling, center etc.
%         Nmcs: The number of Monte Carlo sampling. 
%+++ Output: Structural data: CV
%+++ Hongdong Li, Oct. 16, 2008.
%+++ Revised on Dec.3, 2009.

if nargin<7;OPT=1;end;
if nargin<6;Nmcs=50;end;
if nargin<5;method='center';end;
if nargin<4;K=10;end;
if nargin<3;A=2;end;


[Mx,Nx]=size(X);
A=min([Mx Nx A]);
yytest=[];
yp=[];
nLV=[];
RMSEP=[];
predError=[];
for group=1:Nmcs
    
    DCV=plsdcv(X,y,A,K,method,0,0);
   
    nLV=[nLV;DCV.nLV];
    RMSEP=[RMSEP;DCV.RMSEP];
    predError=[predError;DCV.predError];
    
    if OPT==1;fprintf('The %d/%dth outer loop finished.\n',group,Nmcs);end;
end

%+++ output
  
  RDCV.method=method;
  RDCV.nLV=nLV;
  RDCV.predError=predError;
  RDCV.RMSEP=RMSEP;
  

  