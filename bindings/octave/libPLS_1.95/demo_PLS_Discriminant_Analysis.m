%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  This script is used to test whether the functions in this  %%%%%%%%%%  
%%%%%%%%%%  package can run smoothly.                                  %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%  H.D. Li, lhdcsu@gmail.com                                  %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%+++ Import data;
load DM2;  % a type 2 diabetes data
Xcal=pretreat(Xcal,'autoscaling');
%+++ Cross validation
A=6;
K=5;
method='autoscaling';
N=500;
Nmcs=50;
CV=plsldacv(Xcal,ycal,A,K,method)
MCCV=plsldamccv(Xcal,ycal,A,N,0.6,'autoscaling',0)

%+++ Build a PLS-LDA model
nLV=3;
LDA=plslda(Xcal,ycal,nLV);

%+++ roc curve
roccurve(LDA.yfit,ycal)
 
%+++ different types of Scores plot
figure;
plotlda(LDA,1,0,[2 3 1]);
figure;
plotlda(LDA,1,1,[1  2 3]);
figure;
plotlda(LDA,0,0,[1 2]);
figure;
plotlda(LDA,0,1,[1 2]);
figure;
plotlda(LDA,1,1,[2 3 ]);
figure;
plotlda(LDA,2,1,[3 1]);

%+++ CARS-PLSLDA for variable selection
CARS=carsplslda(Xcal,ycal,A,K,50,method,1); % original version with results not being able to be exactly reproducible
figure;
plotcars_plslda(CARS);

CARS1=carsplslda(Xcal,ycal,A,K,50,method,0); % simplified version with random elements removed so that results can be exactly reproducible
figure;
plotcars_plslda(CARS1);


%+++ MCUVE-PLSLDA for vairbale selection
UVE=mcuveplslda(Xcal,ycal,A,500,0.6,'autoscaling',0);
figure;
bar(UVE.RI,'b','edgecolor','w');
xlim([0 41]);

%+++ SPA for vairable selection: based on Model Population Analysis
N=500;
Q=15;
K=3;
ratio=0.7;
SPA=spa(Xcal,ycal,A,K,Q,N,ratio,method,0);
figure;
bar(SPA.COSS,'b','edgecolor','w');
xlabel('variable index');
xlim([0 41]);
ylabel('COSS');
title('Variable Importance Plot');
figure;
plotspa(SPA,SPA.RankedVariable(1));  
p=SPA.p(SPA.RankedVariable(1))  % the p-value of a given variable


%+++ Random frog: here N is set to 1000 Only for testing.
%    N usually needs to be large, e.g. 10000 considering the huge variable
%    space.
F=randomfrog_plslda(Xcal,ycal,8,'autoscaling',1000,2,0);
bar(F.probability,'b','edgecolor','w');
xlabel('variable index');
ylabel('selection probability');
xlim([0 41]);

%+++ the PHADIA algorithm
load DM2;
F=phadia(Xcal,ycal,5,'autoscaling',5000,8)
figure;
plotphadia(F,0,[35 7 12]); % plot the phase diagram
figure;  
plotphadiavar(F,12);   % plot the distributions of prediction errors of the 249th gene/probe set.
%+++ Test ended

