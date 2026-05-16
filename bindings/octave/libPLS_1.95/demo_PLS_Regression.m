%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  This script is used to test whether the functions in this  %%%%%%%%%%  
%%%%%%%%%%  package can run smoothly.                                  %%%%%%%%%%
%%%%%%%%%%  Suggest: every time you make some modification of codes    %%%%%%%%%%
%%%%%%%%%%  this pakcage, please run this script to debug. Notice that %%%%%%%%%%
%%%%%%%%%%  this script is only for testing so model parameters may not %%%%%%%%%
%%%%%%%%%%  be optimal.                                                %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  H.D. Li, lhdcsu@gmail.com                                  %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ Cross validation
load corn_m51;
A=6;
K=5;
method='center';
N=500;
Nmcs=50;
CV=plscv(X,y,A,K,method)
MCCV=plsmccv(X,y,A,method,N)
DCV=plsdcv(X,y,A,K,method,Nmcs)
RDCV=plsrdcv(X,y,A,K,method,Nmcs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ build a model and make predictions on test set.
load corn_m51;
Rank=ks(X); %+++ Data partition using Kennard-Stone algorithm
Xcal=X(Rank(1:60),:);
ycal=y(Rank(1:60),:);
Xtest=X(Rank(61:80),:);
ytest=y(Rank(61:80),:);
PLS=pls(Xcal,ycal,10);  %+++ Build a PLS regression model using training set
[ypred,RMSEP]=plsval(PLS,Xtest,ytest); %+++ make predictions on test set
figure;
plot(ytest,ypred,'.',ytest,ytest,'r-');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ Outlier detection
load corn_m51;
F=mcs(X,y,12,'center',1000,0.7)
figure;
plotmcs(F);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ CARS-PLS for variable selection
load corn_m51;
A=10;
K=5; 
N=50;
CARS=carspls(X,y,A,K,method,N,0,1);  % original version of CARS
figure;
plotcars(CARS);
CARS1=carspls(X,y,A,K,method,N,1,1);  % original version of CARS but with optimal LV selection using the Standard deviation information.
figure;
plotcars(CARS1);
CARS2=carspls(X,y,A,K,method,N,0,0);  % exactly reproducible version of CARS with random elements removed
figure;
plotcars(CARS2);
CARS3=carspls(X,y,A,K,method,N,1,0);  % exactly reproducible version of CARS but with optimal LV selection using the Standard deviation information.
figure;
plotcars(CARS3);


%+++ moving window PLS
[WP,RMSEF]=mwpls(X,y);
figure;
plot(WP,RMSEF);
xlabel('wavelength');
ylabel('RMSEF');

%+++ Random frog: here N is set to 1000 Only for testing.
%    N usually needs to be large, e.g. 10000 considering the huge variable
%    space.
Frog=randomfrog_pls(X,y,A,method,1000,5);
figure;
plot(Frog.probability);
xlabel('variable index');
ylabel('selection probability');




