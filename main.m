close all
neurons = [3 2];
t = 0:0.01:5;
input = sin(t);
target = sin(t);

epochs = 20;
%p = randperm(length(input),20);

L = NNGenerator( neurons, input(1:end-1), target(1:end-1),...
                              epochs, @RMSE,...
                              @tgh, @linearAct);
                          
                          
