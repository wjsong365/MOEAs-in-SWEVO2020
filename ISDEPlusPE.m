function ISDEPlusPE(Global)
% <algorithm> <0>
% MOEA-ISDE+-PE
% ISDE+ source code:
% https://www.ntu.edu.sg/home/epnsugan/index_files/publications.htm
% ISDE+ reference:
% T. Pamulapati, R. Mallipeddi and P. N. Suganthan,
% "$I_{\rm SDE}$ +—An Indicator for Multi and Many-Objective Optimization,"
% in IEEE Transactions on Evolutionary Computation, vol. 23, no. 2, pp. 346-352, April 2019.

    %% Generate random population
    lastPop = Global.Initialization();
    currPop = Global.Initialization();
    [currPop,Fitness,~] = EnvironmentalSelection([currPop,lastPop],Global.N);
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

    Next = false(1, length(Population));
    
    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Fitness = FrontNo; 
    
    %% ISDE+
    Last     = find(FrontNo==MaxFNo);
    LastFrontPop = Population(Last);
    DistanceValue = F_distance(LastFrontPop.objs);
    Fitness(Last) = Fitness(Last) + DistanceValue;
    [~,Rank] = sort(Fitness,'ascend');
    Next(Rank(1:N)) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    Fitness    = Fitness(Next);
end


%% F_distance
function DistanceValue = F_distance(FunctionValue)
[N,M] = size(FunctionValue);
PopObj = FunctionValue;
%%  sum of OBJECTIVE
fmax   = repmat(max(PopObj,[],1),N,1);
fmin   = repmat(min(PopObj,[],1),N,1);
PopObj = (PopObj-fmin)./(fmax-fmin);
fpr    = mean(PopObj,2);
[~,rank] = sort(fpr); 

%%%%%%%%%%%%% % SDE with Sum of Objectives  %%%%%%%%%%%%%%%%%%%%
DistanceValue = zeros(1,N);
for j = 2 : N
    
    SFunctionValue = max(PopObj(rank(1:j-1),:),repmat(PopObj(rank(j),:),(j-1),1));
    
    Distance = inf(1,j-1);
    
    for i = 1 : (j-1)
        Distance(i) = norm(SFunctionValue(i,:)-PopObj(rank(j),:))/M;
    end
    
    Distance = min(Distance);
    
    DistanceValue(rank(j)) = exp(-Distance);

end
end


%% MatingSelection
function MatingPool = MatingSelection(DistanceValue)

    N = length(DistanceValue);

    %% Binary tournament selection
    Parent1   = randi(N,1,N);
    Parent2   = randi(N,1,N);
    MatingPool = zeros(1,N);
   
    for i = 1:N
       if DistanceValue(Parent1(i)) < DistanceValue(Parent2(i))
           MatingPool(i) = Parent1(i);
       elseif DistanceValue(Parent1(i)) > DistanceValue(Parent2(i))
           MatingPool(i) = Parent2(i);
       else
           if rand< 0.5
                MatingPool(i) = Parent1(i);
           else
                MatingPool(i) = Parent2(i);
           end
       end
    end
end