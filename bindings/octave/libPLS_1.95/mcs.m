function F=mcs(X,y,A,method,N,ratio)
%+++ Monte-Carlo Sampling method for outlier detection.
%+++ X: sample matrix: M x P.
%+++ y: Response variable: p-dimensional.
%+++ A: number of PLS components.
%+++ N: Number of Monte Carlo Sampling, default value:2500.
%+++ ratio: The ratio of samples randomly selected to build a PLS model,
%           default 0.75.
%+++ Ref: Q.S. Xu, Y.Z. Liang, 2001.Chemo Lab,1-11.
%+++ Edited by H.D. Li,Feb.7, 2009.
%+++ Tutor: Yizeng Liang, yizeng_liang@263.net.
%+++ E-mail: lhdcsu@gmail.com.


%+++ Default parametrical settings.
if nargin<6;ratio=0.75;end;
if nargin<5;N=1000;end;
if nargin<4;method='center';end;
if nargin<3;A=3;end;

[Mx,Nx]=size(X);    %  Data size.
A=min([size(X) A]); %  Latent Variable correction.
nc=floor(Mx*ratio); %  Number of calibration samples.
nv=Mx-nc;           %  Number of test samples.
Resi=nan(N,Mx);   %  Initialize the error matrix which records the prediction errors of each sample in each iteration.


%+++ Main loop for calculate the prediction errors for each sample.
for i=1:N
    index=randperm(Mx);
    calk=index(1:nc);testk=index(nc+1:Mx);
    Xcal=X(calk,:);ycal=y(calk);
    Xtest=X(testk,:);ytest=y(testk);        %+++ Randomly select calibration and test set.
    
 
    %   data pretreatment
    [Xs,xpara1,xpara2]=pretreat(Xcal,method);
    [ys,ypara1,ypara2]=pretreat(ycal,'center');   
    
       
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [B,W,T,P,Q]=plsnipals(Xs,ys,A); 
  
     %+++ calculate the coefficient linking Xcal and ycal.
    
     C=ypara2*B./xpara2';
     coef=[C;ypara1-xpara1*C;];    
     
     %+++ predict the test samples
     Xteste=[Xtest ones(size(Xtest,1),1)];
     ytest_p=Xteste*coef;
     pre_resi=ytest_p-ytest;
     Resi(i,testk)=pre_resi';  %+++ Record the prediction errors.
     if mod(i,100)==0;fprintf('The %d/%dth MCS finished.\n',i,N);end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MEAN=zeros(1,Mx);  %+++ Initialie the mean of the prediction errors of each sample.
STD=zeros(1,Mx);   %+++ Initialie the standard deviation of the prediction errors of each sample.
for i=1:Mx;
  resi=Resi(:,i);
  k=find(isnan(resi)==0);  %+++ Find the prediction errors (not exactly equal to zero).
  resi=resi(k);
  MEAN(i)=abs(mean(resi));   %+++ Calculate the mean of prediction errors for each sample.
  STD(i)=std(resi);          %+++ Calculate the standard deviation of prediction errors for each sample.
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ output  %%%%%%%%%%%%%%%%
F.pretreat=method;
F.MCS_parameter=[N ratio];
F.predError=Resi;
F.MEAN=MEAN;
F.STD=STD;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    +++ END  +++   %%%%%%%
