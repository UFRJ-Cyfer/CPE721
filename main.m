close all

neurons = [4];
t = 5:0.1:10;
input = t;
target = sin(t);
% 
csi = 0.9;          % momentum 
eta = [4e-3 4e-3];  %learning rate for BIAS and WEIGHT
epoch = 10;
span = 0.2;
runs = 5e3;
% 
% sstd = std(input);
% smean = mean(input);
% 
% input = X;
input = (input-mean(input))/std(input);
target = (target-mean(target))/std(target);

% [A] Data

% C = [0 0 1 1 ; 0 1 0 1];
% rand('state',0)
% X = []; P = 1;
% for k = 1:size(C,2), 
%     X = [X 0.7*rand(2,P)+repmat(C(:,k),1,P)]; 
% end;
% X = X - repmat(mean(X,2),1,size(X,2));
% plot(X(1,:),X(2,:),'k.');
% t = []; T = [1 -1 -1 1]; 
% for k = 1:size(C,2), 
%     t = [t repmat(T(k),1,P)]; 
% end;
% 
% % [A1] Randomize Data Order
% 
% randn('state',0);
% X = [X ; t ; randn(1,size(X,2))]';
% X = sortrows(X,4)';
% t = X(3,:); X = X(1:2,:);
% 
% rand('state',0)

%p = randperm(length(input),20);

L = NNGenerator( neurons, span, 1, 1);
                          
  [L_train, J] =  trainBPBatch(L, input, target, epoch,runs, csi, eta,...
                @RMSE, @tgh, @linearAct);
 
 plotNNResults(L_train , J ,input ,target, @tgh, @linearAct)
     
%        K = 2;
%        hold on;
% for n = 1:size(X,2),
%     L_train(1).input = X(:,n);
%     for k = 1:K,
%         L_train(k).u = L_train(k).weight*L_train(k).input + L_train(k).bias;
%         L_train(k).o = tanh(L_train(k).u);
%         L_train(k+1).input = L_train(k).o;
%     end;
%     if L_train(K).o > 0, plot(X(1,n),X(2,n),'ko'); end;    
% end;
% figure; plot(J);
       
                          
