function HypEPE(Global)
% <algorithm> <0>
% HypE-PE
% nSample --- 10000 --- Number of sampled points for HV estimation

    %% Parameter setting
    nSample = Global.ParameterSet(10000);

    %% Generate random population
    lastPop = Global.Initialization();
    currPop = Global.Initialization();
    % Reference point for hypervolume calculation
    RefPoint = zeros(1,Global.M) + max(currPop.objs)*1.2;
    [currPop,FrontNo,~] = EnvironmentalSelection([currPop,lastPop],Global.N,RefPoint,nSample);
    
    %% Optimization
    alpha = 1; omega =1.3;
    while Global.NotTermination(currPop)
        Offspring  = PE(lastPop,currPop,FrontNo,Global.N,{alpha, 0.002,0.5,1,20});
        lastPop = currPop;
        [currPop,FrontNo,Next] = EnvironmentalSelection([currPop,Offspring],Global.N,RefPoint,nSample);
        b=sum(Next);
        a=sum(Next(length(currPop)+1:end));
        if a/b > 0.4
            alpha = min(alpha*omega,50);
        else
            alpha = max(alpha/omega,0.1);
        end
    end
end


function [Population,FrontNo,Next] = EnvironmentalSelection(Population,N,RefPoint,nSample)
% The environmental selection of HypE

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last   = find(FrontNo==MaxFNo);
    Choose = true(1,length(Last));
    while sum(Choose) > N-sum(Next)
        drawnow();
        Remain  = find(Choose);
        F       = CalHV(Population(Last(Remain)).objs,RefPoint,sum(Choose)-N+sum(Next),nSample);
        [~,del] = min(F);
        Choose(Remain(del)) = false;
    end
    Next(Last(Choose)) = true;
    % Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
end


function F = CalHV(points,bounds,k,nSample)
% Calculate the hypervolume-based fitness value of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is modified from the code in
% http://www.tik.ee.ethz.ch/sop/download/supplementary/hype/

    [N,M] = size(points);
    if M > 2
        % Use the estimated method for three or more objectives
        alpha = zeros(1,N); 
        for i = 1 : k 
            alpha(i) = prod((k-[1:i-1])./(N-[1:i-1]))./i; 
        end
        Fmin = min(points,[],1);
        S    = unifrnd(repmat(Fmin,nSample,1),repmat(bounds,nSample,1));
        PdS  = false(N,nSample);
        dS   = zeros(1,nSample);
        for i = 1 : N
            x        = sum(repmat(points(i,:),nSample,1)-S<=0,2) == M;
            PdS(i,x) = true;
            dS(x)    = dS(x) + 1;
        end
        F = zeros(1,N);
        for i = 1 : N
            F(i) = sum(alpha(dS(PdS(i,:))));
        end
        F = F.*prod(bounds-Fmin)/nSample;
    else
        % Use the accurate method for two objectives
        pvec  = 1:size(points,1);
        alpha = zeros(1,k);
        for i = 1 : k 
            j = 1:i-1; 
            alpha(i) = prod((k-j)./(N-j))./i;
        end
        F = hypesub(N,points,M,bounds,pvec,alpha,k);
    end
end

function h = hypesub(l,A,M,bounds,pvec,alpha,k)
% The recursive function for the accurate method

    h     = zeros(1,l); 
    [S,i] = sortrows(A,M); 
    pvec  = pvec(i); 
    for i = 1 : size(S,1) 
        if i < size(S,1) 
            extrusion = S(i+1,M) - S(i,M); 
        else
            extrusion = bounds(M) - S(i,M);
        end
        if M == 1
            if i > k
                break; 
            end
            if alpha >= 0
                h(pvec(1:i)) = h(pvec(1:i)) + extrusion*alpha(i); 
            end
        elseif extrusion > 0
            h = h + extrusion*hypesub(l,S(1:i,:),M-1,bounds,pvec(1:i),alpha,k); 
        end
    end
end