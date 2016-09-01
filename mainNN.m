

neurons = [13];

csi = 0;          % momentum
eta = 1e-1*[1 1];  %learning rate for BIAS and WEIGHT
span = 0.2;
runs = 8e3;
lambda = 0;

% [input,target] = extractDataSets('F:\BitBucket\ProjetoFinal\projetofinal\FES2CHMAIN24_V85\Teste\Antigo\',...
%                         'Adriano1204PIRele_4Ciclos.txt');

[inputData,targetData] = extractDataSets('D:\Luiz\projetofinal\FES2CHMAIN24_V85\Teste\Antigo\',...
    'Adriano1204PIRele_4Ciclos.txt');
L_train = NNGenerator( neurons, span, inputData(1).Pulse', 1);



for k=1:length(targetData)
    
    input = inputData(k).Pulse';
    
    for i=1:size(input,1)
        input(k,:) = (input(k,:) - mean(input(k,:)))/std(input(k,:));
    end
    
    target = targetData(k).Pulse';
    target = (target-mean(target))/std(target);
    
    
    
    if k < length(targetData)
        valInput = inputData(k+1).Pulse';
        for i=1:size(input,1)
            valInput(k,:) = (valInput(k,:) - mean(valInput(k,:)))/std(valInput(k,:));
        end
        
        valTarget = targetData(k+1).Pulse';
        valTarget = (valTarget-mean(valTarget))/std(valTarget);
        
    else
        valInput = inputData(1).Pulse';
        for i=1:size(input,1)
            valInput(k,:) = (valInput(k,:) - mean(valInput(k,:)))/std(valInput(k,:));
        end
        
        valTarget = targetData(1).Pulse';
        valTarget = (valTarget-mean(valTarget))/std(valTarget);
    end
    % L = NNGenerator( neurons, span, input, 1);
    
    [L_train, J,J_v] =  trainResBPBatch(L_train, input, target,...
        valInput,valTarget,...
        runs,...
        csi, eta,...
        @RMSE, @tgh, @linearAct);
    
    plotNNResults(L_train , J ,J_v, input ,target, @tgh, @linearAct)
    
end