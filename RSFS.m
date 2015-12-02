function [W] = RSFS(fea,L,Z,Y,alpha,beta,nu,maxIter)
%RSFS: Robust Spectral Learning for Unsupervised Feature Selection
%
%   version 1.0 Oct/2014   
%
%   written by Lei Shi (leishics@gmail.com)


gamma = 10^8;
[nSmp, nFea] = size(fea);
nClass = size(Y,2);

L_plus = (abs(L)+L)/2;
L_minus = (abs(L)-L)/2;

iter = 0;
D = eye(nFea,nFea);
In = eye(nSmp,nSmp);
W = zeros(nFea,size(Y,2));
obj_old = getObj(L,Y,fea,Z,W,alpha,beta,nu,gamma);
disp(['obj',num2str(iter),'=',num2str(obj_old)]);

while iter<maxIter
%update W and D
    W = (fea'*fea+beta/alpha*D)^(-1)*fea'*(Y-Z);
    D = diag(1./sqrt(sum(W.*W,2)+eps))*0.5;
    
%update Z
    Y_tmp = Y-fea*W;
    Z = sign(Y_tmp).*max(abs(Y_tmp)-nu/(2*alpha),0);
    
%update Y
    A = fea*W+Z;
    A_plus = (abs(A)+A)/2;
    A_minus = (abs(A)-A)/2;
    
    term_1 = L_minus*Y+gamma*Y+alpha*A_plus+eps;
    term_2 = L_plus*Y+alpha*Y+gamma*Y*Y'*Y+alpha*A_minus+eps;
    
    Y = Y.*sqrt(term_1./term_2);
    Y = Y*diag(sqrt(1./(diag(Y'*Y)+eps)));
    
    iter = iter+1;
    obj_new = getObj(L,Y,fea,Z,W,alpha,beta,nu,gamma);
    rate = (obj_old-obj_new)/obj_new;
    disp(['obj',num2str(iter),'=',num2str(obj_new),',',num2str(rate)]);
    if rate<10^(-3)
        break;
    else
        obj_old = obj_new;
    end
end
end

function [obj] = getObj(L,Y,X,Z,W,alpha,beta,nu,gamma)
%min Tr(YLY) + alpha * ||Y-XW-Z||_{F,2} + beta * ||W||_{2,1} + nu *
%||Z||_{1} + (gamma/2) * ||YY-I||_{2,1}
Term_1 = Y-X*W-Z;
Ic = eye(size(Y,2),size(Y,2));
obj = trace(Y'*L*Y) + alpha*sum(sum(Term_1.*Term_1))+beta*L21Norm(W)+nu*L1Norm(Z)+0.5*gamma*L21Norm(Y'*Y-Ic);
end


function l1_norm = L1Norm(X)
l1_norm = 0;
nRow = size(X,1);
for i = 1:nRow
    l1_norm = l1_norm + norm(X(i,:),1);
end
end

function l21_norm = L21Norm(X)
l21_norm = 0;
nRow = size(X,1);
for i = 1:nRow
    l21_norm = l21_norm + norm(X(i,:),2);
end
end