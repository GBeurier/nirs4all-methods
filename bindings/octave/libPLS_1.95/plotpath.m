function plotpath(path)
%+++ Plot the model path computed by modelpath.m
%+++ Central South University, Changsha 410083, P.R. China.
%+++ Advisor: Prof. Yizeng Liang, yizeng_liang@263.net.
%+++ Coder: Hongdong Li, Dec. 25, 2009, lhdcsu@gmail.com.


shift=(max(path(:,1))-min(path(:,1)))/50;
plot(path(:,1),path(:,2),'bo','markersize',10);
hold on;
plot(path(:,1),path(:,2),'r-','linewidth',2.5);
text(path(:,1)+shift,path(:,2),num2str([1:size(path,1)]'));