function  [cm,cv]=ConsistentCostMatrixGen(NumClass,maxvalue,CD)

% To generate consistent cost matrix used in the experiments with the following
%   constraints:
%   (1) cost(i,j) = 0 for i=j,
%   (2) at least one cost is 1.0. (union condition)
%   (3) the cost of misclassifying the smallest class to the largest class is the biggest,
%        while the cost of misclassifying the largest class to the smallest class is the
%        smallest. (monotone condition)

% Input:
%   NumClass: number of classes
%   maxvalue: maximum value of the cost matrix to be generated
%   CD: a vector indicate number of instances in each class
% Output:
%   cm: consistent cost matrix, cm(i,j) is the cost of classifying class
%          j-th example to class-i.
%   cv: cost vector for consistent cost matrix cm

% Copyright: Xu-Ying Liu and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)

%
% example:
%
% import matlab.engine
% eng = matlab.engine.start_matlab()
% cmcv = eng.eval('ConsistentCostMatrixGen(3,10,[5, 15,60])',nargout=2)
% cm = np.array(cmcv[0])
% cv = np.array(cmcv[1])
% eng.quit()
%

cv=rand(1,NumClass)*(maxvalue-1)+1;
cv=cv/min(cv);

cv=sort(cv);

cm=zeros(NumClass,NumClass)*Inf;
for i=2:NumClass
    for j=1:i-1
        cm(i,j)=rand(1,1)*(maxvalue-1)+1;
        cm(j,i)=cm(i,j)*cv(i)/cv(j);
    end
end

% confirm condition 1&2
cm=cm/nanmin(nanmin(cm));
for i=1:NumClass
    cm(i,i)=0;
end

% confirm condition 3
[tmp id]=sort(CD,'descend');
tmp=id(2:end-1);
rn=randperm(length(tmp));
id(2:end-1)=tmp(rn);
tmp=[id' [1:NumClass]'];
tmp=sortrows(tmp,1);
id=tmp(:,2)';


cv=cv(id);
cm=cm(id,id);

% check if cv is the root of cm
m=coMatrix(cm);
mm=m(:,2:end);
b=-m(:,1);
cv_=[1; mm\b];
cv_=cv_/min(cv_);
if(sum(cv'-cv_)>1e-6)
    err('wrong.\n')
end

% check if condition 3 meets
[tmp mincostid]=min(cv);
[tmp maxcostid]=max(cv);
if(CD(mincostid)~=max(CD) || CD(maxcostid)~=min(CD))
    error('is not postive consistent cost matrix')
end

fprintf('')


