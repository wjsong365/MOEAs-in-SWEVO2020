function ARMOEAPE(Global)
% <algorithm> <0>
% AR-MOEA-PE

    %% Generate the sampling points and random population
    lastPop = Global.Initialization();
    currPop = Global.Initialization();
    W = UniformPoint(Global.N,Global.M);
    [Archive,RefPoint,Range] = UpdateRefPoint(lastPop(all(lastPop.cons<=0,2)).objs,W,[]);
    [Archive,RefPoint,Range] = UpdateRefPoint([Archive;currPop(all(currPop.cons<=0,2)).objs],W,Range);
    [currPop,Range,FrontNo,~] = EnvironmentalSelection([lastPop,currPop],RefPoint,Range,Global.N);

	alpha = 1; omega =1.3;
    %% Start the interations
    while Global.NotTermination(currPop)
        Offspring  = PE(lastPop,currPop,FrontNo,Global.N,{alpha, 0.002,0.5,1,20});
        lastPop = currPop;
        
        [Archive,RefPoint,Range] = UpdateRefPoint([Archive;Offspring(all(Offspring.cons<=0,2)).objs],W,Range);
        [currPop,Range,FrontNo,Next] = EnvironmentalSelection([currPop,Offspring],RefPoint,Range,Global.N);
        
        b=sum(Next);
        a=sum(Next(length(currPop)+1:end));
        if a/b > 0.4
            alpha = min(alpha*omega,50);
        else
            alpha = max(alpha/omega,0.1);
        end
    end
end

function [Population,Range,FrontNo,Next] = EnvironmentalSelection(Population,RefPoint,Range,N)
% The environmental selection of AR-MOEA

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    CV = sum(max(0,Population.cons),2);
    if sum(CV==0) > N
        %% Selection among feasible solutions
        Population = Population(CV==0);
        % Non-dominated sorting
        [FrontNo,MaxFNo] = NDSort(Population.objs,N);
        Next = FrontNo < MaxFNo;
        % Select the solutions in the last front
        Last   = find(FrontNo==MaxFNo);
        Choose = LastSelection(Population(Last).objs,RefPoint,Range,N-sum(Next));
        Next(Last(Choose)) = true;
        Population = Population(Next);
        % Update the range for normalization
        Range(2,:) = max(Population.objs,[],1);
        Range(2,Range(2,:)-Range(1,:)<1e-6) = 1;
        
        FrontNo = FrontNo(Next); % 
        
    else
        %% Selection including infeasible solutions
        [~,rank]   = sort(CV);
        Population = Population(rank(1:N));
    end
end

function Remain = LastSelection(PopObj,RefPoint,Range,K)
% Select part of the solutions in the last front

    N  = size(PopObj,1);
    NR = size(RefPoint,1);

    %% Calculate the distance between each solution and point
    Distance    = CalDistance(PopObj-repmat(Range(1,:),N,1),RefPoint);
    Convergence = min(Distance,[],2);
    
    %% Delete the solution which has the smallest metric contribution one by one
    [dis,rank] = sort(Distance,1);
    Remain     = true(1,N);
    while sum(Remain) > K
        % Calculate the fitness of noncontributing solutions
        Noncontributing = Remain;
        Noncontributing(rank(1,:)) = false;
        METRIC = sum(dis(1,:)) + sum(Convergence(Noncontributing));
        Metric = inf(1,N);
        Metric(Noncontributing) = METRIC - Convergence(Noncontributing);
        % Calculate the fitness of contributing solutions
        for p = find(Remain & ~Noncontributing)
            temp = rank(1,:) == p;
            noncontributing = false(1,N);
            noncontributing(rank(2,temp)) = true;
            noncontributing = noncontributing & Noncontributing;
            Metric(p) = METRIC - sum(dis(1,temp)) + sum(dis(2,temp)) - sum(Convergence(noncontributing));
        end
        % Delete the worst solution and update the variables
        [~,del] = min(Metric);
        temp    = rank ~= del;
        dis     = reshape(dis(temp),sum(Remain)-1,NR);
        rank    = reshape(rank(temp),sum(Remain)-1,NR);
        Remain(del) = false;
    end
end

function MatingPool = MatingSelection(Population,RefPoint,Range)
% The mating selection of AR-MOEA

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Calculate the degree of violation of each solution
    CV = sum(max(0,Population.cons),2);

    %% Calculate the fitness of each feasible solution based on IGD-NS
    if sum(CV==0) > 1
        % Calculate the distance between each solution and point
        N = sum(CV==0);
        Distance    = CalDistance(Population(CV==0).objs-repmat(Range(1,:),N,1),RefPoint);
        Convergence = min(Distance,[],2);
        [dis,rank]  = sort(Distance,1);
        % Calculate the fitness of noncontributing solutions
        Noncontributing = true(1,N);
        Noncontributing(rank(1,:)) = false;
        METRIC   = sum(dis(1,:)) + sum(Convergence(Noncontributing));
        fitness  = inf(1,N);
        fitness(Noncontributing) = METRIC - Convergence(Noncontributing);
        % Calculate the fitness of contributing solutions
        for p = find(~Noncontributing)
            temp = rank(1,:) == p;
            noncontributing = false(1,N);
            noncontributing(rank(2,temp)) = true;
            noncontributing = noncontributing & Noncontributing;
            fitness(p) = METRIC - sum(dis(1,temp)) + sum(dis(2,temp)) - sum(Convergence(noncontributing));
        end
    else
        fitness = zeros(1,sum(CV==0));
    end

    %% Combine the fitness of feasible solutions with the fitness of infeasible solutions
    Fitness = -inf(1,length(Population));
    Fitness(CV==0) = fitness;
    
    %% Binary tournament selection
    MatingPool = TournamentSelection(2,ceil(length(Population)/2)*2,CV,-Fitness);
end

function [Archive,RefPoint,Range] = UpdateRefPoint(Archive,W,Range)
% Reference point adaption

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

	%% Delete duplicated and dominated solutions
    Archive = unique(Archive(NDSort(Archive,1)==1,:),'rows');
    NA      = size(Archive,1);
    NW      = size(W,1);
    
	%% Update the ideal point
    if ~isempty(Range)
        Range(1,:) = min([Range(1,:);Archive],[],1);
    elseif ~isempty(Archive)
        Range = [min(Archive,[],1);max(Archive,[],1)];
    end
    
    %% Update archive and reference points
    if size(Archive,1) <= 1
        RefPoint = W;
    else
        %% Find contributing solutions and valid weight vectors
        tArchive = Archive - repmat(Range(1,:),NA,1);
        W        = W.*repmat(Range(2,:)-Range(1,:),NW,1);
        Distance      = CalDistance(tArchive,W);
        [~,nearestP]  = min(Distance,[],1);
        ContributingS = unique(nearestP);
        [~,nearestW]  = min(Distance,[],2);
        ValidW        = unique(nearestW(ContributingS));

        %% Update archive
        Choose = ismember(1:NA,ContributingS);
        Cosine = 1 - pdist2(tArchive,tArchive,'cosine');
        Cosine(logical(eye(size(Cosine,1)))) = 0;
        while sum(Choose) < min(3*NW,size(tArchive,1))
            unSelected = find(~Choose);
            [~,x]      = min(max(Cosine(~Choose,Choose),[],2));
            Choose(unSelected(x)) = true;
        end
        Archive  = Archive(Choose,:);
        tArchive = tArchive(Choose,:);

        %% Update reference points
        RefPoint = [W(ValidW,:);tArchive];
        Choose   = [true(1,length(ValidW)),false(1,size(tArchive,1))];
        Cosine   = 1 - pdist2(RefPoint,RefPoint,'cosine');
        Cosine(logical(eye(size(Cosine,1)))) = 0;
        while sum(Choose) < min(NW,size(RefPoint,1))
            Selected = find(~Choose);
            [~,x]    = min(max(Cosine(~Choose,Choose),[],2));
            Choose(Selected(x)) = true;
        end
        RefPoint = RefPoint(Choose,:);
    end 
end

function Distance = CalDistance(PopObj,RefPoint)
% Calculate the distance between each solution to each adjusted reference
% point

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    N  = size(PopObj,1);
    NR = size(RefPoint,1);

    %% Adjust the location of each reference point
    Cosine = 1 - pdist2(PopObj,RefPoint,'cosine');
    NormR  = sqrt(sum(RefPoint.^2,2));
    NormP  = sqrt(sum(PopObj.^2,2));
    d1     = repmat(NormP,1,NR).*Cosine;
    d2     = repmat(NormP,1,NR).*sqrt(1-Cosine.^2);
    [~,nearest] = min(d2,[],1);
    RefPoint    = RefPoint.*repmat(d1(N.*(0:NR-1)+nearest)'./NormR,1,size(RefPoint,2));
    
    %% Calculate the distance between each solution to each point
    Distance = pdist2(PopObj,RefPoint);
end