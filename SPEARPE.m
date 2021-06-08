function SPEARPE(Global)
% <algorithm> <0>
% SPEA/R-PE

%------------------------------- SPEA/R --------------------------------
% S. Jiang and S. Yang, A strength Pareto evolutionary algorithm based on
% reference direction for multiobjective and many-objective optimization,
% IEEE Transactions on Evolutionary Computation, 2017, 21(3): 329-346.
%--------------------------------------------------------------------------

    %% Generate the reference directions (general approach)
    [W,Global.N] = UniformPoint(Global.N,Global.M);
    % Largest acute angle between two neighbouring reference directions
    cosine = 1 - pdist2(W,W,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    theta  = max(min(acos(cosine),[],2));
    
    %% Generate random population
    lastPop = Global.Initialization();
    currPop = Global.Initialization();
    QObj       = ObjectiveNormalization([currPop,lastPop]);
    [Ei,Angle] = Associate(QObj,W);
    FV         = FitnessAssignment(Ei,QObj,Angle,theta);
    [currPop,Fitness,~] = EnvironmentalSelection(Global,[currPop,lastPop],Ei,FV);
    [~,I]=sort(Fitness,2, 'ascend');
    FrontNo=2*ones(1,Global.N);
    FrontNo(I(1:floor(Global.N/2)))=1; 

    %% Optimization
    alpha = 1; omega =1.3; 
    while Global.NotTermination(currPop)
        Offspring  = PE(lastPop,currPop,FrontNo,Global.N,{alpha, 0.002,0.5,1,20});
        lastPop    = currPop;
        QObj       = ObjectiveNormalization([currPop,Offspring]);
        [Ei,Angle] = Associate(QObj,W);
        FV         = FitnessAssignment(Ei,QObj,Angle,theta);
        [currPop,Fitness,Next] = EnvironmentalSelection(Global,[currPop,Offspring],Ei,FV);
        [~,I] = sort(Fitness,2, 'ascend');
        FrontNo = 2*ones(1,Global.N);
        FrontNo(I(1:floor(Global.N/2))) = 1;
        b = sum(Next);
        a = sum(Next(length(currPop)+1:end));
        if a/b > 0.4
            alpha = min(alpha*omega,50);
        else
            alpha = max(alpha/omega,0.1);
        end
    end
end


%% EnvironmentalSelection
function [Population,Fitness,Next] = EnvironmentalSelection(Global,Population,Ei,FV)
% The environmental selection of SPEA/R

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Choose = [];
    while length(Choose) < Global.N
        H = [];
        for i = unique(Ei)
            if i > 0
                Local = find(Ei==i);
                [~,q] = min(FV(Local));
                H     = [H,Local(q)];
            end
        end
        if Global.evaluated >= Global.evaluation
            if any(FV(H)<1)
                H = H(FV(H)<1);
            end
        end
        Ei(H) = -1;
        if length(Choose) + length(H) <= Global.N
            Choose = [Choose, H];       
        else       
            [~,rank] = sort(FV(H));
            Choose   = [Choose,H(rank(1:Global.N-length(Choose)))];
        end
    end
    Population = Population(Choose);
    Fitness    = FV(Choose);
    Next       = false(1,length(FV));
    Next(Choose) = true;
end



%% MatingSelection
function MatingPool = MatingSelection(PopObj,K)
% The mating selection of SPEA/R

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N = size(PopObj,1);
    
    %% The Euclidean distance between each two solutions
    Dis = pdist2(PopObj,PopObj);
    Dis(logical(eye(N))) = inf;
    
    %% Randomly select one solution for each solution
    MatingPool = zeros(1,N);
    for i = 1 : N
        Candidates    = randperm(N,min(K,N));
        [~,nearest]   = min(Dis(i,Candidates));
        MatingPool(i) = Candidates(nearest);
    end
end

%% ObjectiveNormalization
function PopObj = ObjectiveNormalization(Population)
% Objective normalization in SPEA/R

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj = Population.objs;
    ND     = NDSort(PopObj,1) == 1;
    zmin   = min(PopObj(ND,:),[],1);
    zmax   = max(PopObj(ND,:),[],1);
    PopObj = (PopObj-repmat(zmin,size(PopObj,1),1))./repmat(zmax-zmin,size(PopObj,1),1);
end

%% Associate
function [Ei,Angle] = Associate(PopObj,W)
% Associate each solution with a reference direction

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    Angle      = acos(1-pdist2(PopObj,W,'cosine'));
    [Angle,Ei] = min(Angle',[],1);
end

%% FitnessAssignment
function FV = FitnessAssignment(Ei,PopObj,Angle,theta)
% Calculate the local fitness value of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
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
    
    %% Local strength and local raw fitness
    Sl = zeros(1,N);
    Rl = zeros(1,N);
    for i = unique(Ei)
        Local     = find(Ei==i);
        Sl(Local) = sum(Dominate(Local,Local),2);
        for j = Local
            Rl(j) = sum(Sl(Local(Dominate(Local,i))));
        end
    end
    
    %% Global strength and local raw fitness
    Sg = sum(Dominate,2);
    Rg = zeros(1,N);
    for i = 1 : N
        Rg(i) = sum(Sg(Dominate(:,i)));
    end
    
    %% Density
    D = Angle./(Angle+theta);
    
    %% Final fitness
    FV = zeros(1,N);
    for i = unique(Ei)
        Local = find(Ei==i);
        if length(Local) == 1
            FV(Local) = Rl(Local) + D(Local);
        else
            FV(Local) = Rl(Local) + D(Local) + Rg(Local);
        end
    end
end