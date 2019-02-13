function m=coMatrix(cm)

% generate co-efficient matrix for a cost matrix

% Input:
%   cm: c-by-c cost matrix
% Output:
%   m: c*(c-1)/2-by-c co-efficient matrix

% Copyright: Xu-Ying Liu and Zhi-Hua Zhou, 2009
% Contact: Xu-Ying Liu (liuxy@lamda.nju.edu.cn)

N=size(cm,1);

m=zeros(N*(N-1)/2,N);
row=1;
for i=1:N-1
    for j=i+1:N
        m(row, i)=cm(i, j);
        m(row, j)=-cm(j, i);
        row=row+1;
    end
end