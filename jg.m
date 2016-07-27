% 070626 gabriel@pads.ufrj.br (2D)

close all; clear all;

% [A] Data

C = [0 0 1 1 ; 0 1 0 1];
rand('state',0)
X = []; P = 1;
for k = 1:size(C,2), 
    X = [X 0.7*rand(2,P)+repmat(C(:,k),1,P)]; 
end;
X = X - repmat(mean(X,2),1,size(X,2));
plot(X(1,:),X(2,:),'k.');
t = []; T = [1 -1 -1 1]; 
for k = 1:size(C,2), 
    t = [t repmat(T(k),1,P)]; 
end;

% [A1] Randomize Data Order

randn('state',0);
X = [X ; t ; randn(1,size(X,2))]';
X = sortrows(X,4)';
t = X(3,:); X = X(1:2,:);

% [B] Network Init

% [B1] Parameters

K = 2; % Number of Layers
eta = 1; % Learning Rate Initial Value
Delta = 1e-6; % Stop Criterion
N = size(X,2); % Number of Input Vectors
E = 4*P; % Number of Feed-Forward Iterations per Epoch
alpha = 0.99; % Learning Rate Decay Factor

eta = 0.7;
alpha = 1;

% [B2] Layers

L(1).W = rand(2,2)-0.5;
L(1).b = rand(2,1)-0.5;
L(2).W = rand(1,2)-0.5;
L(2).b = rand(1,1)-0.5;

% [C] Batch Error Backpropagation Training

n=1; i=1; fim=0;
while not(fim),

    for k=1:K,
        L(k).db = zeros(size(L(k).b));
        L(k).dW = zeros(size(L(k).W));
    end;
    J(i) = 0;
    for ep=1:E,

        % [C1] Feed-Forward
        
        L(1).x = X(:,n);
        for k = 1:K,
            L(k).u = L(k).W*L(k).x + L(k).b;
            L(k).o = tanh(L(k).u);
            L(k+1).x = L(k).o;
        end;
        e = t(n) - L(K).o;
        J(i) = J(i) + (e'*e)/2;

        % [C2] Error Backpropagation

        L(K+1).alpha = e; L(K+1).W = eye(length(e));
        for k = fliplr(1:K),
            L(k).M = eye(length(L(k).o)) - diag(L(k).o)^2;
            L(k).alpha = L(k).M*L(k+1).W'*L(k+1).alpha;
            L(k).db = L(k).db + L(k).alpha;
            L(k).dW = L(k).dW + kron(L(k).x',L(k).alpha);
        end;
        n = n+1; if n>N, n=1; end;

    end;

    % [C3] Updates

    for k = 1:K,
        L(k).b = L(k).b + eta*L(k).db;
        L(k).W = L(k).W + eta*L(k).dW;
    end;
    J(i) = J(i)/E;

    % [C4] Stop criterion

    if (i>1),
        if (abs(J(i)-J(i-1))/J(i) < Delta)|(i>1000),
            fim = 1;
        end;
    end;
    if not(fim)
        i = i+1; if n>N, n=1; end; eta = eta*alpha;
    end;

end;

% [D] Test

hold on;
for n = 1:size(X,2),
    L(1).x = X(:,n);
    for k = 1:K,
        L(k).u = L(k).W*L(k).x + L(k).b;
        L(k).o = tanh(L(k).u);
        L(k+1).x = L(k).o;
    end;
    if L(K).o < 0, plot(X(1,n),X(2,n),'ko'); end;    
end;
figure; plot(J);