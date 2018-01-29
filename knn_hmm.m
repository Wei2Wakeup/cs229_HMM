%%approach hmm
clc;
clear;
%% load data
load -ascii handwriting.data;
Y = handwriting(:,1);
Y=Y+1;
[coeff,score,latent] = pca(handwriting(:,3:end));
X = score(:,1:40);
word_tot=[Y,handwriting(:,2),X];
%% split word
word_index=find(handwriting(:,2)==1);
N_word=size(word_index,1);

for i=1:N_word
    if i==1
    word{i}=word_tot(1:word_index(i),:);
    else
    word{i}=word_tot(word_index(i-1)+1:word_index(i),:);   
    end
end
%%
Trans_M=zeros(26,26);
start_arr=zeros(26,1);
for i=1:N_word
    word_tmp=word{i}(:,1);
    start_arr(word_tmp(1))=start_arr(word_tmp(1))+1;
    for j=1:size(word_tmp,1)-1
        Trans_M(word_tmp(j),word_tmp(j+1))=Trans_M(word_tmp(j),word_tmp(j+1))+1;
    end
end
base_trans=repmat(sum(Trans_M,2),1,26);
Trans_Prob=Trans_M./base_trans;
start_Prob=start_arr./sum(start_arr);
%% 10 fold
Indices = crossvalind('Kfold', N_word, 10);
Emis_M=zeros(26,26);
test_err_tmp=0;
for cross_i=1:10
%%
%cross_i=1;
test_index=(Indices==cross_i);
test_data=cell2mat(word(test_index)');
test_X=test_data(:,3:end);
test_Y=test_data(:,1);
train_word=word(~test_index)';
train_data=cell2mat(word(~test_index)');
train_X=train_data(:,3:end);
train_Y=train_data(:,1);

    mdl = fitcknn(train_X,train_Y);
    test_Y_hat= predict(mdl,test_X); 
    
    Y_pair=[test_Y,test_Y_hat];
    for j=1:size(Y_pair,1)
    Emis_M(Y_pair(j,1),Y_pair(j,2))=Emis_M(Y_pair(j,1),Y_pair(j,2))+1;
    end
end

base_emis=repmat(sum(Emis_M,2),1,26);
Emis_Prob=Emis_M./base_emis;
%sum(test_Y_hat~=test_Y)/size(test_Y,1)

%%
EMIS_HAT = [zeros(1,size(Emis_Prob,2)); Emis_Prob];

TRANS_HAT = [0 start_Prob'; zeros(size(Trans_Prob,1),1) Trans_Prob];

mdl = fitcknn(X,Y);
filename = 'knn_hmm.mat';
save(filename,'mdl','EMIS_HAT','TRANS_HAT','coeff')