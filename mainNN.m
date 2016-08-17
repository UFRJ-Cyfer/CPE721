neurons = [10];

csi = 0.8;          % momentum 
eta = [1.5 1.5]*1e-2;  %learning rate for BIAS and WEIGHT
epoch = 10;
span = 0.2;
runs = 4e3;
lambda = 0.4;

[input,target] = extractDataSets('F:\BitBucket\ProjetoFinal\projetofinal\FES2CHMAIN24_V85\Teste\Antigo\',...
                        'Adriano1204PIRele_4Ciclos.txt');
input = input(2:end,:);
target = (target - mean(target))/std(target);
                    
for k=1:size(input,1)-2;
   input(k,:) = (input(k,:)-mean(input(k,:)))/std(input(k,:));
end

input(end,:) = input(end,:)/9;
input(end-1,:) = input(end-1,:)/9;


L = NNGenerator( neurons, span, input, 1);
                          
  [L_train, J] =  trainBPBatch(L, input, target, epoch,runs,...
                csi, eta, lambda,...
                @RMSE, @tgh, @linearAct);
 
 plotNNResults(L_train , J ,input ,target, @tgh, @linearAct)