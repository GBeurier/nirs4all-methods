function F=prcurve(score,class,nbin)
%+++ function:calculate roc curve with each bin contain roughly the same
%             number of true positives.
%+++ input:  x---a matrix with two colums: C1:score. C2: class
%+++ output: R---a 4-column matrix: [tp,fp,tn,fn],in order.
%+++ Oct. 30, 2012
%+++ Hongdong Li, Ann Arbor

if nargin<3;nbin=1000;end

[score,index]=sort(score);
class(class==-1)=0;
class=class(index);
ntp=sum(class);
interval=floor(ntp/nbin);
t=ceil(linspace(1,ntp-interval,nbin));
cumtp=cumsum(class);
R=nan(nbin,5);
for i=1:length(t)
  k=find(cumtp<=t(i));
  k=k(end);
  cutoff=score(k);
  class1=class(k+1:end); 
  class0=class(1:k); 
  tp=sum(class1);
  fp=sum(class1==0);
  tn=sum(class0==0);
  fn=sum(class0);
  R(i,:)=[tp fp tn fn cutoff];
end

tp=R(:,1);
fp=R(:,2);
tn=R(:,3);
fn=R(:,4);

recall=tp./(tp+fn);
precision=tp./(tp+fp);
auprc=abs(trapz(recall,precision));
F.R=R;
F.recall=recall;
F.precision=precision;
F.auprc=auprc;
