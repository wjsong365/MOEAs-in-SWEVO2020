function SPEA2PE(Global)
% <algorithm>  <0>
% SPEA2-PE

    %% Generate random population
    lastPop = Global.Initialization();
    currPop = Global.Initialization();
    [currPop,Fitness,~] = EnvironmentalSelection([lastPop,currPop],Global.N);
    [~,I]=sort(Fitness,2, 'ascend');
    FrontNo=2*ones(1,Global.N);
    FrontNo(I(1:floor(Global.N/2)))=1;
    
    %% Optimization
    alpha = 1; omega =1.3; 
    while Global.NotTermination(currPop)
        Offspring  = PE(lastPop,currPop,FrontNo,Global.N,{alpha, 0.002,0.5,1,20});
        lastPop = currPop;
        [currPop,Fitness,Next] = EnvironmentalSelection([currPop,Offspring],Global.N);
        [~,I]=sort(Fitness,2, 'ascend');
        FrontNo=2*ones(1,Global.N);
        FrontNo(I(1:floor(Global.N/2)))=1;
        temp = zeros(1,length([currPop,Offspring]));
        temp(Next) = 1;
        Next = temp;
        b=sum(Next);
        a=sum(Next(length(currPop)+1:end));
        if a/b > 0.4
            alpha = min(alpha*omega,50);
        else
            alpha = max(alpha/omega,0.1);
        end
    end
    
end

function [Population,Fitness,Next] = EnvironmentalSelection(Population,N)
% The environmental selection of SPEA2

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the fitness of each solution
    Fitness = CalFitness(Population.objs);

    %% Environmental selection
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    % Population for next generation
    Population = Population(Next);
    Fitness    = Fitness(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end

function Fitness = CalFitness(PopObj)
% Calculate the fitness of each solution

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);

    %% Detect the dominance relation between each two solutions
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end
    
    %% Calculate S(i)
    S = sum(Dominate,2);
    
    %% Calculate R(i)
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    
    %% Calculate D(i)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    
    %% Calculate the fitnesses
    Fitness = R + D';
end