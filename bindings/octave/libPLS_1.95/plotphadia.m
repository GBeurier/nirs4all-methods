function plotphadia(F,flag,label)
%+++ Grouping Variable Projection based on Model Population Analysis.
%+++ Originally proposed by Prof. Yizeng liang&Hongdong Li, yizeng_liang@263.net.
%+++ Implemented by Hongdong Li, lhdcsu@gmail.com.
%+++ Jul.19, 2009, Changsha City, China.
%+++ 

if nargin<3;label=[];end
if nargin<2;flag=0;end

variableIndex=[1:length(F.DMEAN)]';
if size(label,1)==1;label=label';end
%+++ Get DMEAN and DSD resulting from GVP algorithm.
DMEAN=F.DMEAN;
DSD=F.DSD;
k0=intersect(find(F.p<0.05),find(DMEAN>0));
k2=intersect(find(F.p<0.05),find(DMEAN<0));
k1=find(F.p>=0.05);

shift=(max(DMEAN)-min(DMEAN))/100; %+++ make the mark not overlapped.
hold on;
if flag==0 
    h=plot(DMEAN(k1),DSD(k1),'b.','markersize',10); 
    h=plot(DMEAN(k0),DSD(k0),'g.','markersize',10); 
    h=plot(DMEAN(k2),DSD(k2),'r.','markersize',10); 
elseif flag==1 
    for j=1:length(k1)
        text(DMEAN(k1(j)),DSD(k1(j)),num2str(variableIndex(k1(j))),'color','b');
    end
    for j=1:length(k0)
        text(DMEAN(k0(j)),DSD(k0(j)),num2str(variableIndex(k0(j))),'color','g');
    end
   
    for j=1:length(k2)
        text(DMEAN(k2(j)),DSD(k2(j)),num2str(variableIndex(k2(j))),'color','r');
    end
end
% label
if length(label)>0
    
    text(DMEAN(label)+shift,DSD(label),num2str(label));

end



%+++ Region partition
Vline=linspace(min(DSD),max(DSD),50);
Hline=linspace(min(DMEAN),max(DMEAN),50);
plot(zeros(1,50),Vline,'k-');
plot(Hline,zeros(1,50),'k-');
d1=(max(DMEAN)-min(DMEAN))/10;
d2=(max(DSD)-min(DSD))/15;
axis([min(DMEAN)-d1,max(DMEAN)+2*d1 min(DSD)-d2 max(DSD)+d2]);
xlabel('DMEAN');ylabel('DSD');
box on;


