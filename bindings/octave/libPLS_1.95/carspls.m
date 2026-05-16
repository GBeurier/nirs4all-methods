function CARS=carspls(X,y,A,fold,method,num,selectLV,originalVersion,order) 
%+++ CARS: Competitive Adaptive Reweighted Sampling method for variable selection.
%+++ X: The data matrix of size m x p
%+++ y: The reponse vector of size m x 1
%+++ A: the maximal principle to extract.
%+++ fold: the group number for cross validation.
%+++ method: pretreatment method.
%+++ num: the  number of Monte Carlo Sampling runs.
%+++ selectLV:     0: selecting the optimal LV achieving the global minimum of RMSECV curves
%                  1: selecting the optimal LV achieving the maximum RMSECV within the range of global minimum + 1 standard deviation.
%+++ originalVersion:  1: using the original version of CARS which
%                         by design may give results which can not be
%                         exactly reproduced due to the embeding random
%                         sampling technique but the results can be
%                         statistically reproduced if you run CARS multiple
%                         times, say 50 times.This is default.
%                      0: using a simplified version of CARS with the random
%                         sampling element removed. So each run of CARS
%                         will give exactly the same results, which is also an                       a
%                         appreciated fature though "containing randomness"
%                         is the nature of data. Considering the 'exact'
%                         reproducibility, this simplified version is used
%                         as default. You can set this value to 1 if want
%                         to use the original version
%+++ order:  0: samples are sorted based on y values
%            1: samples are randomly ordered
%            2: samples are in the same order as in the orginal input
%+++ Hongdong Li, Dec.15, 2008.
%+++ Advisor: Yizeng Liang, yizeng_liang@263.net
%+++ lhdcsu@gmail.com
%+++ Ref:  Hongdong Li, Yizeng Liang, Qingsong Xu, Dongsheng Cao, Key
%    wavelengths screening using competitive adaptive reweighted sampling 
%    method for multivariate calibration, Anal Chim Acta 2009, 648(1):77-84

tic;
%+++ Initial settings.
if nargin<9;order=0;end;
if nargin<8;originalVersion=0;end;
if nargin<7;selectLV=1;end;
if nargin<6;num=50;end;
if nargin<5;method='center';end;
if nargin<4;fold=5;end;
if nargin<3;A=2;end;

%+++ Initial settings.
[Mx,Nx]=size(X);
A=min([Mx Nx A]);
index=1:Nx;
ratio=0.9;
r0=1;
r1=2/Nx;
Vsel=1:Nx;
Q=floor(Mx*ratio);
W=zeros(Nx,num);
Ratio=zeros(1,num);

%+++ Parameter of exponentially decreasing function. 
b=log(r0/r1)/(num-1);  a=r0*exp(b);

%+++ Main Loop
for iter=1:num
     
     if originalVersion == 1
     perm=randperm(Mx);   
     Xcal=X(perm(1:Q),:); ycal=y(perm(1:Q));   %+++ Monte-Carlo Sampling.
     else
     Xcal=X;ycal=y;    
     end
     PLS=pls(Xcal(:,Vsel),ycal,A,method);    %+++ PLS model
     w=zeros(Nx,1);coef=PLS.regcoef_original(1:end-1,end);
     w(Vsel)=coef;W(:,iter)=w; 
     w=abs(w);                                  %+++ weights
     [ws,indexw]=sort(-w);                      %+++ sort weights
     
     ratio=a*exp(-b*(iter+1));                      %+++ Ratio of retained variables.
     Ratio(iter)=ratio;
     K=round(Nx*ratio);  
     
     w(indexw(K+1:end))=0;                          %+++ Eliminate some variables with small coefficients.  
     
     if originalVersion == 1
     Vsel=unique(randsample(Nx,Nx,true,w));                 %+++ Reweighted Sampling from the pool of retained variables.                 
     else              
     Vsel=unique(find(w~=0));
     end
     fprintf('The %dth variable sampling finished.\n',iter);    %+++ Screen output.
 end

%+++  Cross-Validation to choose an optimal subset;
RMSECV=zeros(1,num);
Q2_max=zeros(1,num);
LV=zeros(1,num);
for i=1:num
   vsel=find(W(:,i)~=0);
 
   CV=plscv(X(:,vsel),y,A,fold,method,0,order);  
   if selectLV == 0
   RMSECV(i)=CV.RMSECV_min;
   Q2_max(i)=CV.Q2_max;   
   LV(i)=CV.optLV;
   elseif selectLV==1
   RMSECV(i)=CV.RMSECV_min_1SD;
   Q2_max(i)=CV.Q2_max_1SD;   
   LV(i)=CV.optLV_1SD; 
   end
   
   fprintf('The %d/%dth subset finished.\n',i,num);
end
[RMSECV_min,indexOPT]=min(RMSECV);
Q2_max=max(Q2_max);




%+++ save results;
time=toc;
%+++ output
CARS.W=W;
CARS.time=time;
CARS.RMSECV=RMSECV;
CARS.RMSECV_min=RMSECV_min;
CARS.Q2_max=Q2_max;
CARS.iterOPT=indexOPT;
CARS.optLV=LV(indexOPT);
CARS.ratio=Ratio;
CARS.vsel=find(W(:,indexOPT)~=0)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




