close all
neurons = [2];
t = 0:0.1:5;
input = sin(t);
target = sin(t);

epochs = 2;
    
input = (input-mean(input))/std(input);
target = (target-mean(target))/std(target);

%p = randperm(length(input),20);

L = NNGenerator( neurons, input(1:end-1), target(1:end-1),...
                              epochs, @RMSE,...
                              @tgh, @linearAct);
                          
 [L_train, J] =  trainBPBatch(L,input,target,epochs,...
                @RMSE, @tgh,@linearAct);
 
 plotNNResults(L_train , J ,input,target, @tgh, @linearAct)
                          
                          
