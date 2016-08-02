function [ ] = plotNNResults( L,J, input, target, ...
    inner_activation, outter_activation)
%PLOTNNRESULTS Summary of this function goes here
%   Detailed explanation goes here

neurons = L.neurons;

err = zeros(size(target));
output = zeros(size(input)); % caso simples de aproximador universal

    for m=1:length(input)
        [output(m),~] = feedforward(L,input(:,m),...
            inner_activation,outter_activation);
        if m < length(target)
            err(m) = target(:,m) - output(m);
        end
    end
    t = 5:0.1:10;
    ax = sin(t);
    
    teste = 0:0.1:15;
    ax_= sin(teste);
    
    subplot(3,1,1)
    
    hold on;
%   plot(input*sstd+smean,'b');
    plot(input*std(t)+mean(t),output*std(ax)+mean(ax),'r'); 
    plot(t,ax,'ko');  
%   legend('NN Input','NN Output','Target');
    legend('NN Output','Target');
    
    subplot(3,1,2)
    hold on
    plot(err*std(ax),'r')
%   plot(input*sstd+smean - (output*std(ax)+mean(ax)+0.05)','k');
%   legend('Error clean sinc','error noised sinc');
    
    subplot(3,1,3)
    plot(J);
    legend('Cost');
end

