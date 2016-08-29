% [input,target] = extractDataSets('F:\BitBucket\ProjetoFinal\projetofinal\FES2CHMAIN24_V85\Teste\Antigo\',...
%                         'Adriano1204PIRele_4Ciclos.txt');
% input = input(2:end,:);
% 
% target = (target - mean(target))/std(target);
% target = (target-min(target));
%                     
% for k=1:size(input,1);
%    input(k,:) = (input(k,:)-mean(input(k,:)))/std(input(k,:));
% end

net = feedforwardnet([13],'trainrp');
net = configure(net, input, target);
net.trainParam.epochs = 8000;
[net,tr] = train(net, input, target);

a = net(input);
plot(a)
hold on
plot(target)
