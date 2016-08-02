function [ ] = plotNNResults( L,J, input, target, ...
    inner_activation, outter_activation, sstd, smean)
%PLOTNNRESULTS Summary of this function goes here
%   Detailed explanation goes here
    if isrow(input)
        input = input';
    end
    
    if isrow(target)
        target = target';
    end
neurons = L.neurons;

    for m=1:length(input)
        [output(m),err(m),~] = feedforward(L,input(m,:),target(m,:),...
            inner_activation,outter_activation);
    end
    t = 0:0.1:5;
    ax = sinc(t);
    subplot(3,1,1)
    
    hold on;
    plot(input*sstd+smean,'b');
    plot(output*std(ax)+mean(ax)+0.05,'r'); 
    plot(target*std(ax)+mean(ax),'k');  
    legend('NN Input','NN Output','Target');

    subplot(3,1,2)
    hold on
    plot(err,'r')
    plot(input*sstd+smean - (output*std(ax)+mean(ax)+0.05)','k');
    legend('Error clean sinc','error noised sinc');
    
    subplot(3,1,3)
    plot(J);
    legend('Cost');
    
    
end

