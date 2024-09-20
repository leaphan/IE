function [P Kv  Kx U Z KY_test steps obj] = FCLGSC(X,VFCM,UFCM,tt,ProK, Palpha,Penta,Pgamma,maxStep,conCriterion, ker_type)
%% robust jointly sparse fuzzy C-means
% X:数据集， n*m，行向量形式
% W:由LLE导出的相似度矩阵
% ProK：选择的特征数目
%Pgamma，Pmu, Penta：参数
% maxStep：最大迭代步数
% conCriterion ：收敛条件
%P: m*ProK;
%V cluster_n*m  行向量
%U n*cluster_n  行向量
%InitU  cluster_n*n 

%%--------------------begin------------------------------------------------
data_n = size(X,1);
dim_n = size(X,2);
options = [2, 50, 1e-5, 0];
Ns=size(VFCM,1);
Nt=size(X,1);
I=ones(Nt,Nt);        %全是1的矩阵
IA=eye(Nt);             %单位矩阵
Z =ones(Ns,Nt); 
R1 = zeros(Ns,Nt);
J=zeros(Ns,Nt);
expo = 2;
InitU=UFCM;
mf = InitU.^expo; 
% V = mf*X./((ones(size(X, 2), 1)*sum(mf'))'); % 根据初始的U，计算得到的V
X=X';
V=VFCM';
U = InitU';    %n*cluster_n 
Xt_test=tt';


Kernel='gauss';
[le,lr]=size(X);
X=X(:,randperm(lr));
% Xr=X(:,1:floor(le*(0.5)));


switch Kernel 
       case'linear'    
       kervar1=0.5;% free parameter
       kervar2=10;% no use 
       case  'gauss'
       kervar1=1.2;% free parameter
       kervar2=10;% no use
end  
 %Xr=X(:,1:lr);
Xr=X(:,1:floor(le*(0.5)));
Xr = Xr./repmat(sqrt(sum(Xr.^2)),[size(Xr,1) 1]); 
K  = gram(Xr',Xr',Kernel,kervar1,kervar2);
K =max(K,K');
K = K./repmat(sqrt(sum(K.^2)),[size(K,1) 1]);  
Kx   = gram(Xr',X',Kernel,kervar1,kervar2);
Kx = Kx./repmat(sqrt(sum(Kx.^2)),[size(Kx,1) 1]);  
Kv   = gram(Xr',V',Kernel,kervar1,kervar2);
Kv = Kv./repmat(sqrt(sum(Kv.^2)),[size(Kv,1) 1]);  
KY_test  = gram(Xr',Xt_test',Kernel,kervar1,kervar2);
KY_test = KY_test./repmat(sqrt(sum(KY_test.^2)),[size(KY_test,1) 1]); 

dim_n = size(Kx ,1);
P = eye(dim_n,dim_n);    %初始化为单位阵
mu1=0.1;         %%R更新系数
max_mu = 10^6;
tau=1;         %%MMD系数
theta=1;   %%MC系数
t=0.3;      %%Z梯度下降系数
t1=0.3;      %%Z梯度下降系数
lambdaJ=1;      %%辅助变量J系数
rho =1.01;
 k=2;
%        k2=Graph_Xt_test_num;
 s_pa=1;
 [L,D,W]=Graph_Laplacian(X', k, s_pa);  


steps=0;
converged=false;


while ~converged && steps<=maxStep
     
    steps=steps+1; 
    
%    V_Old = V;
    KV_Old =Kv;

  %%%Update U  
      
     R= (EuDist2((P'*Kx)',(P'*Kv)')).^2;
       
     for i=1:1:data_n
        vi = -1*R(i,:)/(2*Palpha);
        U(i,:) = EProjSimplex(vi);
     end    
    U1=diag( sum(U,1) );
    
%     U1 = U1./repmat(sqrt(sum(U1.^2)),[size(U1,1) 1]);
   %%% Update Z 

       Zh1=2*theta/(Nt^2)*Kv'*P*P'*Kv*Z*D;
       Zh2=2*theta/(Nt^2)*Kv'*P*P'*Kx*W;
       Zh3=R1+mu1*(Z-J);
       Zh4=2*tau/(Nt^2)*Kv'*P*P'*Kv*Z*I;
       Zh5=2*tau/(Nt^2)*Kv'*P*P'*Kx*I;
       Derta_Zold=Zh1-Zh2+Zh3+Zh4-Zh5; 
       Derta= Derta_Zold;
       Z_iter=t*Derta; 
       Z=Z-Z_iter;   
      
   %%% Update J
    ta = lambdaJ/mu1;
    temp_J = Z + R1/mu1;
    [UU,sigma,Vr] = svd(temp_J,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>ta));
    if svp>=1
        sigma = sigma(1:svp)-ta;
    else
        svp = 1;
        sigma = 0;
    end
    J = UU(:,1:svp)*diag(sigma)*Vr(:,1:svp)';
  
 %%%Update V
     V1=2*P*P'*Kv*U1-2*P*P'*Kx*U;
     V2=2*Penta*P*P'*Kv*Z*D*Z'-2*Penta*P*P'*Kx*W*Z';
     V3=2*Penta*1/(Nt^2)*P*P'*Kv*Z*I*Z'-2*Penta*1/(Nt^2)*P*P'*Kx*I*Z';
     Derta_V=V1+V2+V3;
     Kv_iter=t1*Derta_V; 
     Kv=Kv-Kv_iter; 
     Kv = Kv./repmat(sqrt(sum(Kv.^2)),[size(Kv,1) 1]);
%     这里对KV试试归一化
  %%%Update P
  
  ph1=2*Kx*I*Kx'-2*Kv*U'*Kx'-2*Kx*U*Kv'+2*Kv*U1*Kv';
  ph2=2*Kv*Z*D*Z'*Kv'+2*Kx*D*Kx'-2*Kv*Z*W*Kx'-2*Kx*W*Z'*Kv';
  ph3=1/(Nt^2)*Kv*Z*I*Z'*Kv'-1/(Nt^2)*Kv*Z*I*Kx'-1/(Nt^2)*Kx*I*Z'*Kv'+1/(Nt^2)*Kx*I*Kx';
  ph4=K;
  AA=ph1+Penta*ph2+Penta*ph3+Pgamma*ph4;
  AA = AA./repmat(sqrt(sum(AA.^2)),[size(AA,1) 1]);
  AA=real(AA);
%   AA=AA+0.001*eye(size(AA,1));
 [eigvector eigvalue] = eig(AA);    
 eigvalue = diag(eigvalue);            %%从小到大排列
 [junk, index] = sort(eigvalue);       %升序
 eigvalue = eigvalue(index);
 eigvector = eigvector(:, index);
    
%     maxEigValue = max(abs(eigvalue));
%     eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
%     eigvalue (eigIdx) = [];
%     eigvector (:,eigIdx) = [];
    
    if ProK < length(eigvalue)             %取最小的k个特征值对应的特征向量
       eigvalue = eigvalue(1:ProK);
       eigvector = eigvector(:, 1:ProK);
    end
    P = eigvector;
    P=real(P);
    nsmp=size(P,2);   %dim_n*ProK  行向量
    for i=1:nsmp
       P(:,i)=P(:,i)/norm(P(:,i),2);
    end   
 
    %%% Update R1 
    R1 = R1+mu1*(Z-J);         
    %%% updating mu
    mu1 = min(rho*mu1,max_mu);
    
    
    
    %%%% 计算obj
    obj(steps,1)=trace(P'*Kx*I*Kx'*P)-2*trace(P'*Kv*U'*Kx'*P)+trace(P'*Kv*U1*Kv'*P);
    obj(steps,2)=Pgamma*norm(U,'fro');
    obj(steps,3)=Penta*trace(P'*Kv*Z*D*Z'*Kv'*P)+Penta*trace(P'*Kx*D*Kx'*P)-Penta*2*trace(P'*Kv*Z*W*Kx'*P);
    obj(steps,4)=Penta*1/(Nt^2)*norm((P'*(Kv*Z-Kx)*I),2);
    obj(steps,5)=Penta*sum(svd(Z));
    obj(steps,6)=Pgamma*trace(P'*K*P);
    obj(steps,7)=obj(steps,1)+obj(steps,2)+obj(steps,3)+obj(steps,4)+obj(steps,5)+obj(steps,6);
    
%       obj = 0;
    
    %if convergent?
    nsmp=size(Kv,1);   
%     for i=1:nsmp
%        ErrorV(i) = norm( Kv(i,:)-KV_Old(i,:), 2);
%     end 
%     criterion = max( ErrorV );
%     if criterion < conCriterion
%         converged=true;
%     end     
end 

%%--------------------end--------------------------------------------------