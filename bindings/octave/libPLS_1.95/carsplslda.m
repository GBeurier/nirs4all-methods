function F=carsplslda(X,y,A,fold,num,method,originalVersion,weight) 
%+++ CARS: Competitive Adaptive Reweighted Sampling method for variable selection.
%+++ A:   the maximal number of PLS components to extract.
%+++ fold: number of folds for cross validation.
%+++ num: the number of Monte Carlo Sampling.
%+++ method: pretreat method, 'center' or 'autoscaling'.
%+++ weight: 0 or 1
%+++ originalVersion:  1: using the original version of CARS which
%                         by design may give results which can not be
%                         exactly reproduced due to the embeding random
%                         sampling technique but the results can be
%                         statistically reproduced if you run CARS multiple
%                         times, say 50 times.
%                      0: using a simplified version of CARS with the random
%                         sampling element removed. So each run of CARS
%                         will give exactly the same results, which is also an                       a
%                         appreciated fature though "containing randomness"
%                         is the nature of data. Considering the 'exact'
%                         reproducibility, this simplified version is used
%                         as default. You can set this value to 1 if want
%                         to use the original version.
%+++ Advisor: Yizeng Liang, yizeng_liang@263.net.
%+++ Hongdong Li, Jan.3, 2009.
%+++ Reference: Hongdong Li, Yizeng Liang, Qingsong Xu and Dongsheng Cao,
%         Key wavelengths screening using competitive adaptive reweighted sampling method
%         for multivariate calibration. Anal. Chim. Acta, 2009, 648 (1): 77-84 



%+++ Initial settings.
if nargin<8;weight=0;end
if nargin<7;originalVersion=0;end
if nargin<6;method='autoscaling';end
if nargin<5;num=50;end
if nargin<4;fold=2;end
if nargin<3;A=2;end

[Mx,Nx]=size(X);
A=min([Mx Nx A]);
kp=find(y==1);
kn=find(y==-1);
X1=X(kp,:);y1=y(kp);N1=length(kp);
X2=X(kn,:);y2=y(kn);N2=length(kn);
index=1:Nx;
ratio=0.95;
r0=1;
r1=2/Nx;
Vsel=1:Nx;
Q1=floor(N1*ratio);
Q2=floor(N2*ratio);
W=zeros(num,Nx);
Ratio=zeros(1,num);

%+++ Parameter of exponentially decreasing function. 
b=log(r0/r1)/(num-1);  a=r0*exp(b);

%+++ Main Loop
for iter=1:num
     
     perm1=randperm(N1);   
     perm2=randperm(N2);   
     
     if originalVersion == 1
     Xcal=[X1(perm1(1:Q1),:);X2(perm2(1:Q2),:)]; 
     ycal=[y1(perm1(1:Q1));y2(perm2(1:Q2))];    %+++ Monte-Carlo Sampling.
     else
     Xcal=X;
     ycal=y;
     end
     
     LDA=plslda(Xcal(:,Vsel),ycal,A,method,weight);    %+++ PLS model
          
     w=zeros(Nx,1);
     coef=LDA.regcoef_lda_origin(1:end-1);
     w(Vsel)=coef;
     
     W(iter,:)=w'; 
     w=abs(w);                                  %+++ weights
     [ws,indexw]=sort(-w);                      %+++ sort weights
     
     ratio=a*exp(-b*(iter+1));                      %+++ Ratio of retained variables.
     Ratio(iter)=ratio;
     K=round(Nx*ratio);  
     
     
     w(indexw(K+1:end))=0;                      %+++ Eliminate some variables with small coefficients.  
     
     if originalVersion == 1
     Vsel=unique(randsample(Nx,Nx,true,w));                     %+++ Reweighted Sampling from the pool of retained variables.                 
     else
     Vsel=unique(find(w~=0));  
     end
     
     
     fprintf('The %dth variable sampling finished.\n',iter);    %+++ Screen output.
 end

%+++  Cross-Validation to choose an optimal subset;
RMSEP=zeros(1,num);
Rpc=zeros(1,num);
for i=1:num
   vsel=find(W(i,:)~=0);
   CV=plsldacv(X(:,vsel),y,A,fold,method,weight,0);
   RMSEP(i)=min(CV.error);
   Rpc(i)=CV.optLV;
   fprintf('The %dth subset finished.\n',i);
end
Rmin=min(RMSEP);
indexOPT=find(RMSEP==Rmin);
indexOPT=indexOPT(end);

%+++ output
F.method=method;
F.W=W;
F.error=RMSEP;
F.error_min=Rmin;
F.iterOPT=indexOPT;
F.optLV=Rpc(indexOPT);
F.ratio=Ratio;
F.vsel=find(W(indexOPT,:)~=0)';
%+++ END




