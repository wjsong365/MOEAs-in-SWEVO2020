function Offspring = PE(lastPop,currPop,FrontNo,NOff,Parameter)
% Path Evolution (PE)

%------------------------ PE operator -------------------------------------
% W. Song, W. Du, C. Fan, W. Zhong, and F. Qian, "A novel path-based 
% reproduction operator for multi-objective optimization," Swarm and 
% Evolutionary Computation, vol. 59, p. 100741, 2020/12/01/ 2020, 
% doi: https://doi.org/10.1016/j.swevo.2020.100741.
%--------------------------------------------------------------------------

    %% Parameter Setting
    if nargin > 4
        [alpha,minC,CR,proM,disM] = deal(Parameter{:});
    else
        [alpha,minC,CR,proM,disM] = deal(0.3, 0.002, 0.5, 1, 20);
    end
    if isa(lastPop(1),'INDIVIDUAL')
        calObj = true;
        lastPop = lastPop.decs;
        currPop = currPop.decs;
    else
        calObj = false;
    end
    Global = GLOBAL.GetObj();
    [Np,D] = size(lastPop);
    Upper = Global.upper;
    Lower = Global.lower;

    %% Evolution Path
    lastCenter = mean(lastPop);
    currCenter = mean(currPop);
    nep = max(abs((currCenter-lastCenter)./(Upper-Lower)));

    %% Disturbance
    if nep < minC
        xi = randi(D,1);
        currCenter(xi) = rand*(Upper(xi)-Lower(xi)) + Lower(xi);
    end
    
    %% Generate Potential Solutions
    Offspring = zeros(size(currPop));
    currPath = currCenter-lastCenter;
    for i=1:Np
        Offspring(i,:)=alpha*currPath.*rand(size(currCenter)) + currPop(i,:); % 前进
    end

    %% Boundary Handling
    Offspring = min(max(Offspring,Lower),Upper);

    %% Polynomial Mutation
    Offspring = PolynomialMutation(Lower,Upper,Offspring,proM,disM);
    
    %% Gene Sharing
    Offspring = GeneSharing(CR, currPop, Offspring, FrontNo); % 

    Offspring = Offspring(1:NOff,:);
    if calObj
        Offspring = INDIVIDUAL(Offspring);
    end

end


%% Gene Sharing
function NewOffspring = GeneSharing(CR,Parent,Offspring,FrontNo)
[Np,D] = size(Parent);
NewOffspring = zeros(Np, D);

F1 = find(FrontNo==min(FrontNo));
NF1 = length(F1);

for i = 1: Np
    ri = randi(D,1);
    for j = 1: D
        if rand > CR && ri ~= j
            NewOffspring(i,j) = Parent(F1(randi(NF1,1)),j);
        else
            NewOffspring(i,j) = Offspring(i,j);
        end
    end
end
end


%% Polynomial mutation
function OffspringDec = PolynomialMutation(Lower,Upper,ParentDec,proM,disM)
    [N, D]       = size(ParentDec);
    OffspringDec = ParentDec;
    
    %% Polynomial mutation
    Lower = repmat(Lower,N,1);
    Upper = repmat(Upper,N,1);
    Site  = rand(N,D) < proM/D;
    mu    = rand(N,D);
    temp  = Site & mu<=0.5;
    OffspringDec(temp) = OffspringDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                         (1-(OffspringDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    OffspringDec(temp) = OffspringDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                         (1-(Upper(temp)-OffspringDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));     

end

