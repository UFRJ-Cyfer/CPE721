

neurons = [13];

csi = 0;          % momentum 
eta = [0.2 0.2];  %learning rate for BIAS and WEIGHT
span = 0.2;
runs = 8e3;
lambda = 0;

[input,target] = extractDataSets('F:\BitBucket\ProjetoFinal\projetofinal\FES2CHMAIN24_V85\Teste\Antigo\',...
                        'Adriano1204PIRele_4Ciclos.txt');
input = input(2:end,:);


% input = 0:0.01:5;
% target = sinc(input);

target = (target - mean(target))/std(target);
% input = (input - mean(input))/std(input);
                    
[trainInd, valInd,testInd] = dividerand(length(target),0.8,0.1,0.1);


% target = (target - mean(target))/std(target);
% target = (target-min(target));
% 
for k=1:size(input,1);
   input(k,:) = (input(k,:)-mean(input(k,:)))/std(input(k,:));
end

% input = input(trainInd);
% target = target(trainInd);


% input = input(:,trainInd);
% target = target(1,trainInd);


% input(end,:) = input(end,:)/9;
% input(end-1,:) = input(end-1,:)/9;


L = NNGenerator( neurons, span, input, 1);
                          
  [L_train, J] =  trainResBPBatch(L, input, target, epoch,runs,...
                csi, eta, lambda,...
                @RMSE, @tgh, @linearAct);
 
 plotNNResults(L_train , J ,input ,target, @tgh, @linearAct)