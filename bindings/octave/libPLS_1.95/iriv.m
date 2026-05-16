function F=iriv(X,y,A_max,fold,method)
%     IRIV: Iteratively Retaining Informative Variables
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A_max: The max PC for cross-validation
%            fold: the group number for cross validation. when fold = m, it is leave-one-out CV
%            method: pretreatment method. Contains: autoscaling, center etc.

%+++ Yonghuan Yun, June. 2, 2013, yunyonghuan@foxmail.com
%+++ Advisor: Yizeng Liang, yizeng_liang@263.net

%++++Ref:  Yong-Huan Yun, Wei-Ting Wang, Hong-Dong Li, Yizeng Liang, Qingsong Xu, A strategy that iteratively 
%          retains informative variables for selecting optimal variable subset in multivariate calibration, 
%           Anal Chim Acta, 2014. http://dx.doi.org/10.1016/j.aca.2013.11.032

%+++       Hongyan Zhang, Haiyan Wnag. Improving  accuracy for cancer classificationwith a new algorithm for 
%          gene selection.  BMC Bioinformatics, November 2012, 13:298
if nargin<5;method='center';end
if nargin<4;fold=5;end
if nargin<3;A_max=10;end

CV=plscv(X,y,A_max,fold,method,0);
A=CV.optLV;               % Choose the optimal principle components for PLS 
XX=X;
[~,Nx]=size(X);   
time=0; 
tic;
varnumber=1:Nx;
j=1;
remain_number(j)=Nx;
control=0;
while control==0      
    [~,Nx]=size(X);
    % determine the row number of binary matrix  based on the number of retained variables of each round      
    if Nx>=500; row=500; end      
    if Nx>=300&&Nx<500; row=300; end      
    if Nx>=100&&Nx<300; row=200; end       
    if Nx>=50&&Nx<100; row=100; end       
    if Nx>=10&&Nx<50; row=50; end    
    if Nx<10;  row=20;end
    
    RMSECV=zeros(row,1);      
    RMSECV_origin=zeros(row,Nx);      
    RMSECV_replace=zeros(row,Nx);          
    binary_matrix=generate_binary_matrix(Nx, row);      
    RMSECV = get_matrix_result( X,y,binary_matrix,A,fold);     
    RMSECV_origin=repmat(RMSECV,1,Nx);     
    RMSECV_replace = get_replace_result(X,y,binary_matrix,A,fold);                    
    RMSECV_exclude = RMSECV_replace;      
    RMSECV_include = RMSECV_replace;      
    RMSECV_exclude(binary_matrix==0) = RMSECV_origin(binary_matrix==0);       
    RMSECV_include(binary_matrix==1) = RMSECV_origin(binary_matrix==1);          
    DMEAN=zeros(1,Nx);       
    Pvalue=zeros(1,Nx);       
    H=zeros(1,Nx);
    for i=1:Nx()           
        %+++ Mann-Whitney U test for variable assessment           
        [p,h] = ranksum(RMSECV_exclude(:,i),RMSECV_include(:,i),'alpha',0.05);           
        Pvalue(i)=p;H(i)=h;          
        temp_DMEAN=mean(RMSECV_exclude(:,i))-mean(RMSECV_include(:,i));        
        %+++ Just a trick, indicating uninformative and interfering variable if Pvalue>1.         
        if temp_DMEAN<0;Pvalue(i)=p+1;end;                
        DMEAN(i)=temp_DMEAN;      
    end   
    ST{j}=H;      
    strong{j}=intersect(varnumber(H==1),varnumber(Pvalue<1));        
    weak{j}=intersect(varnumber(H==0),varnumber(Pvalue<1));     
    uinformative{j}=intersect(varnumber(H==0),varnumber(Pvalue>1));      
    interfering{j}=intersect(varnumber(H==1),varnumber(Pvalue>1));           
    time=time+toc;      
    remove_variables{j}=varnumber(Pvalue>1);
    store=zeros(2,length(varnumber));
    store(1,:)=varnumber;
    store(2,:)=Pvalue;
    P{j}=store;
    j=j+1; 
    remain_number(j)=sum(Pvalue<1);   
    
    % observe whether there are uninformative and intefering variables or not               
    if sum(Pvalue>=1)>0              
        varnumber(Pvalue>=1)=[]; 
        X=XX(:,varnumber);
        fprintf('The %d th round of IRIV has finished!  ', j-1)
        fprintf('Remain %d / %d  variable, using time: %g seconds!\n', sum((Pvalue<1) ),Nx, time)
    else control=1;       
    end  
end
fprintf('The iterative rounds of IRIV have been finished, now enter into the process of backward elimination!\n')

  % backward elimination after several iterative rounds   
variables = backward_elimination(X,y,A,fold,method,varnumber);     
F.SelectedVariables=variables;
F.Remain_number=remain_number;      % the number of remained variables in each round
F.time=time;
F.Pvalue=P;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%+++ subfunctions
function left_variables = backward_elimination(X,y,A,fold,method,varnumber)

% Inputs:
%+++ X: The data matrix of size m x p
%+++ y: The reponse vector of size m x 1
%+++ A: the maximal principle to extract.
%+++ fold: the group number for cross validation.
%+++ method : pretreatment method.
%
% Outputs:
%   left_variables : the variables that finally left.



CV=plscv(X,y,A,fold,method,0);
RMSECV0 = CV.RMSECV_min;
screenProcess=nan(size(X,2)-1,size(X,2)+2);

jj=1;
di=1:size(X,2);
index=1:size(X,2);
while size(X,2)>1
    parc=size(X,2);
    fprintf('Cross-validation with whole variables of this turn: %g\n',RMSECV0);
    screenProcess(jj,1)=RMSECV0;
    RMSECV_temp=nan(1,parc);    
    i=1;
    for k=1:parc
        cX=X;
        cX(:,k)=[];
        CV1=plscv(cX,y,A,fold,method,0);
        RMSECV_temp(k) = CV1.RMSECV_min;
%         fprintf('Cross-validation  without variables #%g: %g\n',index(k),RMSECV_temp(k));  
        if di(index(k))==0
            i=i+1;
        else
            %   screenProcess: the screening process of backward_elimination
            screenProcess(jj,di(index(k))+i)=RMSECV_temp(k);
        end
    end 
    [minRMSECV,minRMSECVIndex]=min(RMSECV_temp);
    jj=jj+1;
    if minRMSECV<=RMSECV0 
        fprintf('Variable #%g has been washed out, %d variables have been deleted.\n',varnumber(minRMSECVIndex),jj-1);
        screenProcess(jj-1,end)=index(minRMSECVIndex);
        di(index(minRMSECVIndex))=0;
        X(:,minRMSECVIndex)=[];
        index(minRMSECVIndex)=[];
        varnumber(minRMSECVIndex)=[];
        RMSECV0=minRMSECV;
    else
        disp('No any variable-deleting is necessary, screenning is finished.');
        disp(['Left variables: ',sprintf('%g,',varnumber)]);
        break
    end
end

left_variables=varnumber;



function binary_matrix = generate_binary_matrix(X_num, row)
%+++ Input:  X_num: the number of vairbales in the X matrix
%            row: the number of the row of binary matrix
%           
%+++ Output: binary_matrix: the matrix contains 0 and 1.

rand_binary = nan(row, X_num);
control = 0;
while control == 0

    for n = 1:X_num
        rand_row = randperm(row);
        rand_row(rand_row<=length(rand_row)/2) = 0;
        rand_row(rand_row>0) = 1;
        rand_binary(:,n) = rand_row';
    end

    for m = 1:row
        if sum(rand_binary(m,:)) <= 1
            control = 0;
            break; 
        else
            control = 1;
        end
    end

end
rand_binary(rand_binary==0) = 0;
binary_matrix=rand_binary;


function RMSECV = get_matrix_result(X,y,binary_matrix,A,fold)


binary_matrix_row = size(binary_matrix,1);
for m = 1:binary_matrix_row
    temp = binary_matrix(m,:);
    del_X = temp==0;
    X_new = X;
    X_new(:,del_X) = [];
    CV=plscv(X_new,y,A,fold,'center',0);
    RMSECV(m)=CV.RMSECV_min;
%  fprintf('The %d(th) circle 1 has finished, elapsed time is %g seconds!!\n', m, toc)
end
RMSECV=RMSECV';
   

function RMSECV_replace = get_replace_result(X,y,binary_matrix,A,fold)

binary_matrix_row = size(binary_matrix,1);
binary_matrix_col = size(binary_matrix,2);
RMSECV_replace = nan(binary_matrix_row, binary_matrix_col);
tic
for m = 1:binary_matrix_col 
    replace = binary_matrix;
    replace(binary_matrix(:,m)==1,m) = 0;
    replace(binary_matrix(:,m)==0,m) = 1;
    RMSECV_replace(:,m) = get_matrix_result( X,y,replace,A,fold);
    fprintf('The %dth/%d has finished, elapsed time is %g seconds!!\n', m,binary_matrix_col, toc)
end












