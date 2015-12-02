function f = RSFS_single(dataset)
%run RSFS

%======================setup=======================
nKmeans = 20;
FeaNumCandi = [];

alfaCandi = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3];
betaCandi = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3];
nuCandi = [10^-3,10^-2,10^-1,1,10^1,10^2,10^3];

maxIter = 20;
%==================================================

%create a folder named by the name of dataset
if exist(dataset) == 0
    mkdir(dataset);
end

load(dataset);

[nSmp,nFea] = size(fea);
if nFea>=300
    FeaNumCandi = [50,100,150,200,250,300];
else
    FeaNumCandi = [50,80,110,140,170,200];
end

bestNMI_max = zeros(length(FeaNumCandi),1);
bestNMI_sqrt = zeros(length(FeaNumCandi),1);
bestACC = zeros(length(FeaNumCandi),1);

nClass = length(unique(gnd));

%print the setup information
disp(['Dataset: ',dataset]);
disp(['class_num=',num2str(nClass),',','num_kmeans=',num2str(nKmeans)]);

t_start = clock;
%construct the affinity matrix
disp(['construct the affinity matrix...']);
S = constructW(fea); 


disp(['get Laplacian matrix...']);
nRowS = size(S,1);

for i = 1:nRowS
    sum_row = sum(S(i,:));
    S(i,:) = S(i,:)/sum_row;
end

diag_ele_arr = sum(S+S',2);
A = diag(diag_ele_arr);
L = A-S-S';

eY = eigY(L,nClass);
rand('twister',5489);
label = litekmeans(eY,nClass,'Replicates',20);
Y_init = zeros(size(fea,1),nClass);
for i = 1:size(fea,1)
    Y_init(i,label(i)) = 1;
end

%Clustering using selected features
for alpha = alfaCandi
    for beta = betaCandi
        for nu = nuCandi
            Y = Y_init;
            disp(['alpha=',num2str(alpha),',','beta=',num2str(beta),',','nu=',num2str(nu)]);
            result_path = strcat(dataset,'\','alpha_',num2str(alpha),'_beta_',num2str(beta),'_nu_',num2str(nu),'_result.mat');
            mtrResult = [];
            Z = zeros(size(fea,1),length(unique(gnd)));
            W = RSFS(fea,L,Z,Y,alpha,beta,nu,maxIter);

            [dumb idx] = sort(sum(W.*W,2),'descend');
            orderFeature_path = strcat(dataset,'\','feaIdx_','alpha_',num2str(alpha),'_beta_',num2str(beta),'_nu_',num2str(nu),'.mat');
            save(orderFeature_path,'idx');
            
            for feaIdx = 1:length(FeaNumCandi)
                feaNum = FeaNumCandi(feaIdx);
                newfea = fea(:,idx(1:feaNum));
                rand('twister',5489);
                arrNMI_max = zeros(nKmeans,1);
                arrNMI_sqrt = zeros(nKmeans,1);
                arrACC = zeros(nKmeans,1);
                for i = 1:nKmeans
                    label = litekmeans(newfea,nClass,'Replicates',1);
                    arrNMI_max(i) = NMI_max_lei(gnd,label);
                    arrNMI_sqrt(i) = NMI_sqrt_lei(gnd,label);
                    arrACC(i) = ACC_Lei(gnd,label);
                end
                mNMI_max = mean(arrNMI_max);
                sNMI_max = std(arrNMI_max);
                mNMI_sqrt = mean(arrNMI_sqrt);
                sNMI_sqrt = std(arrNMI_sqrt);
                mACC = mean(arrACC);
                sACC = std(arrACC);
                if mNMI_sqrt>bestNMI_sqrt(feaIdx)
                    bestNMI_sqrt(feaIdx) = mNMI_sqrt;
                end
                if mACC > bestACC(feaIdx)
                    bestACC(feaIdx) = mACC;
                end
                if mNMI_max > bestNMI_max(feaIdx)
                    bestNMI_max(feaIdx) = mNMI_max;
                end
                mtrResult = [mtrResult,[feaNum,mNMI_max,sNMI_max,mNMI_sqrt,sNMI_sqrt,mACC,sACC]'];
            end
            save(result_path,'mtrResult');
        end
    end
end
t_end = clock;
disp(['exe time: ',num2str(etime(t_end,t_start))]);

%save the best results among all the parameters
result_path = strcat(dataset,'\','best','_result_',dataset,'_RSFS','.mat');
save(result_path,'FeaNumCandi','bestNMI_sqrt','bestACC','bestNMI_max');

f = 1;
end