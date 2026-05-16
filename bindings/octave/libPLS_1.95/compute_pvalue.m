function p=compute_pvalue(X,y)
%+++ X: n x p
%+++ y: n x 1

for i=1:size(X,2)
    [tmp,pi]=corr(X(:,i),y);
    p(i)=pi;
end



