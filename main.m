close all
neurons = [5];
t = 0:0.1:5;
input = sinc(t)+0.1*rand(size(sinc(t)));
target = sinc(t);
csi = 0.3;
eta = [1e-3 1e-2];

epoch = 5;
    sstd = std(input);
    smean = mean(input);
input = (input-mean(input))/std(input);
target = (target-mean(target))/std(target);

%p = randperm(length(input),20);

L = NNGenerator( neurons, input(1:end-1), target(1:end-1));
                          
 [L_train, J] =  trainBPBatch(L,input,target, epoch, csi, eta,...
                @RMSE, @tgh, @linearAct);
 
 plotNNResults(L_train , J ,input,target, @tgh, @linearAct, sstd, smean)
                          
                          
