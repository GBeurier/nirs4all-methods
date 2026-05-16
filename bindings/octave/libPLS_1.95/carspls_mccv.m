function CV=carspls_mccv(X,y,A,N,method,num,selectLV,originalVersion) 
%+++ Cross validation of CARS, at each iteration, CARS is run on sub-training
%    data and tested on the lef-out test samples. Repeat N times, and
%    record the times each variable is selected, say z. So the selection
%    frequency can then be calculated by freq=z/N. In this way, how sample
%    variation can affect variable selection can be investigated. Those
%    variables selected with high frequency are considered to be more
%    robust to sample variation or noise etc, and hence could be useful to
%    be included for developing a model.
%+++ X: The data matrix of size m x p
%+++ y: The reponse vector of size m x 1
%+++ A: the maximal principle to extract.
%+++ N: the number of Monte Carlo samplings;
%+++ method: pretreatment method.
%+++ num: the  number of Monte Carlo Sampling runs for CARS.
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
%                         is the nature of data.
%+++ Hongdong Li, Dec.15, 2008.
%+++ Advisor: Yizeng Liang, yizeng_liang@263.net
%+++ lhdcsu@gmail.com
%+++ Ref:  Hongdong Li, Yizeng Liang, Qingsong Xu, Dongsheng Cao, Key
%    wavelengths screening using competitive adaptive reweighted sampling 
%    method for multivariate calibration, Anal Chim Acta 2009, 648(1):77-84
%+++ Output:
%         freqMatrix: a N x p matrix recording the selected variables in
%                     each iteration. Frequency can be calculated by simply
%                     summing each column.

tic;
%+++ Initial settings.
if nargin<8;originalVersion=1;end;
if nargin<7;selectLV=1;end;
if nargin<6;num=50;end;
if nargin<5;method='center';end;
if nargin<4;N=100;end;
if nargin<3;A=2;end;

[Mx,Nx]=size(X);
A=min([size(X) A]);
Q=floor(0.8*Mx);
Ytrue=[];
Ypred=[];
vsel_each={};
Q2_each=[];
freqMatrix=zeros(N,Nx);
for group=1:N
    
    nperm=randperm(Mx);
    calk = nperm(1:Q);
    testk = nperm(Q+1:Mx);  
    
    Xcal=X(calk,:);
    ycal=y(calk);
    
    Xtest=X(testk,:);
    ytest=y(testk);
    
    CARS=carspls(Xcal,ycal,A,5,method,num,selectLV,originalVersion);
    Q2_each(group)=CARS.Q2_max;
    vsel_each{group}=CARS.vsel;
    freqMatrix(group,CARS.vsel)=1;
 
    PLS=pls(Xcal(:,CARS.vsel),ycal,CARS.optLV,method);
    ypred=plsval(PLS,Xtest(:,CARS.vsel),ytest);
    
    Ypred=[Ypred;ypred];
    Ytrue=[Ytrue;ytest];
    fprintf('The %dth MC cross validation of CARS done.\n',group); 
end

% combine variable
varunion=[];
for i=1:N
    varunion=[varunion vsel_each{i}];
end
varunion=unique(varunion);

%+++ return the original order
Q2=1-sum((Ypred-Ytrue).^2)/sum((Ytrue-mean(Ytrue)).^2);
RMSECV=sqrt(sum((Ypred-Ytrue).^2)/length(Ytrue));

%+++ output
CV.Q2=Q2;
CV.RMSECV=RMSECV;
CV.Ytrue=Ytrue;
CV.Ypred=Ypred;
CV.vsel=vsel_each;
CV.freqMatrix=freqMatrix; % record the selected variables at each iteration.
CV.var_union=varunion;
CV.Q2_each=Q2_each;




