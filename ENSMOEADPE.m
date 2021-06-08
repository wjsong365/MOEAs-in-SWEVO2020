function ENSMOEADPE(Global)
% <algorithm> <0>
% ENS-MOEA/D-PE


    %% Parameter setting
    % [NS,LP] = Global.ParameterSet(30:30:120,50);
    T = ceil(Global.N/10);
    NS = ceil(T/2):floor(T/2):2*T;
    LP = 10;

    %% Generate the weight vectors
    [W,Global.N] = UniformPoint(Global.N,Global.M);
    % Maximum number of solutions replaced by each offspring
    nr = ceil(Global.N/100);

    %% Detect all the neighbours of each solution
    B = pdist2(W,W);
    [~,B] = sort(B,2);

    %% Generate random population
    lastPopulation = Global.Initialization();
    Population = Global.Initialization();
    Z          = min(Population.objs,[],1);
    % Utility for each subproblem
    Pi = ones(Global.N,1);
    % Old Tchebycheff function value of each solution on its subproblem
    oldObj = max(abs((Population.objs-repmat(Z,Global.N,1)).*W),[],2);

    %% Optimization
    p           = ones(1,length(NS))./length(NS);
    FEs         = zeros(1,length(NS));
    FEs_success = zeros(1,length(NS));
    while Global.NotTermination(Population)
        % Select neighborhood size for each subproblem
        ns = RouletteWheelSelection(Global.N,1./p);
        
        % Apply MOEA/D-DRA for one generation
        for subgeneration = 1 : 5
            % Choose I
            Bounday = find(sum(W<1e-3,2)==Global.M-1)';
            I = [Bounday,TournamentSelection(10,floor(Global.N/5)-length(Bounday),-Pi)];

            % For each solution in I
            for i = I
                % Choose the parents
                if rand < 0.9
                    P = B(i,randperm(NS(ns(i))));
                else
                    P = randperm(Global.N);
                end

                % Generate an offspring
                % Offspring = DE(Population(i),Population(P(1)),Population(P(2)));
                FrontNo = ones(1,length(P));
                Offspring = PE(lastPopulation(P),Population(P),FrontNo,1,{0.3, 0.002,0.5,1,20});

                % Update the ideal point
                Z = min(Z,Offspring.obj);

                % Update the solutions in P by Tchebycheff approach
                g_old   = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                g_new   = max(repmat(abs(Offspring.obj-Z),length(P),1).*W(P,:),[],2);
                replace = find(g_old>=g_new,nr);
                Population(P(replace)) = Offspring;
                if ~isempty(replace)
                    FEs_success(ns(i)) = FEs_success(ns(i)) + 1;
                end
                FEs(ns(i)) = FEs(ns(i)) + 1;
            end
        end
        lastPopulation = Population;
        if ~mod(Global.gen,10)
            % Update Pi for each solution
            newObj    = max(abs((Population.objs-repmat(Z,Global.N,1)).*W),[],2);
            DELTA     = (oldObj-newObj)./oldObj;
            Temp      = DELTA < 0.001;
            Pi(~Temp) = 1;
            Pi(Temp)  = (0.95+0.05*DELTA(Temp)/0.001).*Pi(Temp);
            oldObj    = newObj;
        end
        if ~mod(Global.gen,LP)
            % Update the probability of choosing each neighborhood size
            R           = FEs_success./FEs+0.05; % The small constant value ε = 0.05 is used to avoid the possible zero selection probabilities
            p           = R./sum(R);
            FEs         = zeros(1,length(NS));
            FEs_success = zeros(1,length(NS));
        end
    end
end