

neurons = [5];

csi = 0;          % momentum 
eta = [1 1];  %learning rate for BIAS and WEIGHT
span = 0.3;
runs = 1e5;
lambda = 0;
context = 3;

[input,target] = extractDataSets('F:\BitBucket\ProjetoFinal\projetofinal\FES2CHMAIN24_V85\Teste\Antigo\',...
                        'Adriano1204PIRele_4Ciclos.txt');
input = input(2,:);
dataContext = 0.5*ones(context,size(input,2));

input = cat(1,input,dataContext);


% input = 0:0.01:5;
% target = sinc(input);

target = (target - nanmean(target))/nanstd(target);
target(isnan(target))=0;
% input = (input - mean(input))/std(input);
                    
[trainInd, valInd,testInd] = dividerand(length(target),0.8,0.1,0.1);


% target = (target - mean(target))/std(target);
% target = (target-min(target));
% 
for k=1:size(input,1);
   input(k,:) = (input(k,:)-nanmean(input(k,:)))/nanstd(input(k,:));
end
input(isnan(input))=0;
% input = input(trainInd);
% target = target(trainInd);


% input = input(:,trainInd);
% target = target(1,trainInd);


% input(end,:) = input(end,:)/9;
% input(end-1,:) = input(end-1,:)/9;


L = elmanNNGenerator( neurons, span, context, input, 1);
                          
  [L_train, J] =  elmanNNResBP(L, input, target, context,runs,...
                csi, eta, lambda,...
                @RMSE, @tgh, @linearAct);
 
 plotNNResults(L_train , J ,input ,target, @tgh, @linearAct)