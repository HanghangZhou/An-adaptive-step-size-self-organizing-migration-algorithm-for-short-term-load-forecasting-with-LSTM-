function [RunResult, RunValue, RunTime, RunFES, RunOptimization, RunParameter] = CPD_BBO(problem, N, runmax)
'CPD_BBO'
D = Dim(problem); % 3-6 rows, please refer to CEP
lu = Boundary(problem, D);
TEV = Error();
FESMAX = 10000 * D;
RunOptimization = zeros(runmax, D);

for run = 1 : runmax
    TimeFlag = 0;
    TempFES = FESMAX;
    t1 = clock;
    
    x = Initpop(N, D, lu); % population initialization
    
    fitness = benchmark_func(x, problem); % calculate individual fitness
    FES = N; % current function evaluation times
    [fitness_sorted, order] = sort(fitness); % the population was ordered by fitness
    sort_population = x(order, :);
    
    parent_population = sort_population; %% in order to cumulate population information , the parent_population is used to save the parent of sort_population
    
    % Parameterized initialization
    Iteration = 1; % Current iteration
    OPTIONS.popsize = N; % total population size
    OPTIONS.Maxgen = FESMAX / N; % generation count limit
    OPTIONS.numVar = D; % number of genes in each population member
    OPTIONS.pmodify = 1; % habitat modification probability
    OPTIONS.pmutate = 0.005; % initial mutation probability
    Keep = 2; % elitism parameter: how many of the best habitats to keep from one generation to the next
    lambdaLower = 0.0; % lower bound for migration probability per gene
    lambdaUpper = 1; % upper bound for migration probability per gene
    dt = 1; % step size used for numerical integration of probabilities
    I = 1; % max immigration rate for each island
    E = 1; % max emigration rate for each island
    P = OPTIONS.popsize; % max species count for each island
    for popindex = 1 : OPTIONS.popsize
        Population(popindex).chrom = sort_population(popindex, :);
        Population(popindex).fitness = fitness_sorted(popindex);
    end
    % Initialize the species count probability of each habitat
    % Later we might want to initialize probabilities based on cost
    for j = 1 : length(x)
        Prob(j) = 1 / length(x);
    end
    h = 1; % sampling points during each run
    Pe = 0.5; % devising this parameter to control the ratio of CMM to the original migration
    % Begin the optimization loop 
    while FES <= FESMAX
        % Save the best habitats in a temporary array
        for j = 1 : Keep
            chromKeep(j, :) = Population(j).chrom;
            fitnessKeep(j) = Population(j).fitness;
        end
        % Map fitness values to species counts
        [Population] = GetSpeciesCounts(Population, P);
        % Calculate immigration rate and emigration rate for each species count
        % lambda(i) is the immigration rate for individual i
        % mu(i) is the emigration rate for individual i
        [lambda, mu] = GetLambdaMu(Population, I, E, P);
        
        % ProbFlag = true or false, whether or not to use probabilities to
        % update migration rates and to mutate
        ProbFlag = true;
        if ProbFlag
            % Compute the time derivative of Prob(i) for each habitat i.
            for j = 1 : length(Population)
                % Compute lambda for one less than the species count of habitat i.
                lambdaMinus = I * (1 - (Population(j).SpeciesCount - 1) / P);
                % Compute mu for one more than the species count of habitat i.
                muPlus = E * (Population(j).SpeciesCount + 1) / P;
                % Compute Prob for one less than and one more than the species count of habitat i.
                % Note that species counts are arranged in an order opposite to that presented in
                % MacArthur and Wilson's book - that is, the most fit
                % habitat has index 1, which has the highest species count.
                if j < length(Population)
                    ProbMinus = Prob(j+1);
                else
                    ProbMinus = 0;
                end
                if j > 1
                    ProbPlus = Prob(j-1);
                else
                    ProbPlus = 0;
                end
                ProbDot(j) = -(lambda(j) + mu(j)) * Prob(j) + lambdaMinus * ProbMinus + muPlus * ProbPlus;
            end
            % Compute the new probabilities for each species count.
            Prob = Prob + ProbDot * dt;
            Prob = max(Prob, 0);
            Prob = Prob / sum(Prob);
        end
        % Now use lambda and mu to decide how much information to share between habitats
        lambdaMin = min(lambda);
        lambdaMax = max(lambda);
        if Iteration == 1 % when the Iteration is not 1, the cumulative population distribution information is used
            [Q, ~] = eig(cov(sort_population)); % get the eigenvectors of the covariance matrix
            X_eig =sort_population * Q; % transform the original coordinate system into eigen coordinate system
        else
%             [Q1, ~] = eig(cov(sort_population)); % get the eigenvectors of the covariance matrix
%             X_eig1 =sort_population * Q1; % transform the original coordinate system into eigen coordinate system
%             [Q2, ~] = eig(cov(parent_population));
%             X_eig2 =parent_population * Q2;
%             cNP = 1; % learning rate
%             X_eig = (1 - cn) .* X_eig2 + cn .* X_eig1;

            for i = 1 : OPTIONS.popsize
                w(i) = log(OPTIONS.popsize + 0.5) - log(i);
            end
            for i = 1 : OPTIONS.popsize
                w(i) = w(i) / sum(w);
            end
            
            
            % revision
            m = zeros(1, OPTIONS.numVar);
            for i = 1 : OPTIONS.popsize
                temp = w(i) .* parent_population(i, :);
                m = m + temp;
            end
            estimator = zeros();
            for i = 1 : OPTIONS.popsize
                estimator = estimator + w(i) * (sort_population(i, :) - m) * (sort_population(i, :) - m)';
            end
            
            
            NPeff = 1 / sum(w .* w);
            cNP = min(1, NPeff / OPTIONS.popsize^2);
            cov_population = (1 - cNP) .* cov(parent_population) + cNP * estimator;
            %cov_population = (1 - cNP) .* cov(parent_population) + cNP .* cov(sort_population);
            [Q, ~] = eig(cov_population);
            X_eig = sort_population * Q;
        end
        for k = 1 : length(Population)
            if rand > OPTIONS.pmodify
                continue;
            end
            % Normalize the immigration rate
            lambdaScale = lambdaLower + (lambdaUpper - lambdaLower) * (lambda(k) - lambdaMin) / (lambdaMax - lambdaMin);
            % Probabilistically input new information into habitat i
            % Select migration based covariance matrix or original
            % migration according to parameter Pe
            if rand < Pe
                for j = 1 : OPTIONS.numVar
                    if rand < lambdaScale
                        % Pick a habitat from which to obtain a feature
                        RandomNum = rand * sum(mu);
                        Select = mu(1);
                        SelectIndex = 1;
                        while (RandomNum > Select) && (SelectIndex < OPTIONS.popsize)
                            SelectIndex = SelectIndex + 1;
                            Select = Select + mu(SelectIndex);
                        end
                        Island(k,j) = X_eig(SelectIndex, j);
                    else
                        Island(k,j) = X_eig(k, j);
                    end
                end
                Island(k, :) =Island(k, :) * Q'; 
            else
                for j = 1 : OPTIONS.numVar
                    if rand < lambdaScale
                        % Pick a habitat from which to obtain a feature
                        RandomNum = rand * sum(mu);
                        Select = mu(1);
                        SelectIndex = 1;
                        while (RandomNum > Select) && (SelectIndex < OPTIONS.popsize)
                            SelectIndex = SelectIndex + 1;
                            Select = Select + mu(SelectIndex);
                        end
                        Island(k,j) = Population(SelectIndex).chrom(j);
                    else
                        Island(k,j) = Population(k).chrom(j);
                    end
                end
            end
        end
        if ProbFlag
            % Mutation
            Pmax = max(Prob);
            MutationRate = OPTIONS.pmutate * (1 - Prob / Pmax);
            % Mutate the all of the solutions
            for k = 1 : length(Population)
            % Mutate only the worst half of the solutions          
            % for k = round(length(Population)/2) : length(Population)
                for parnum = 1 : OPTIONS.numVar
                    if MutationRate(k) > rand
                        Island(k,parnum) = floor(lu(1) + (lu(2) - lu(1)) * rand);
                        % Make sure each individual is legal
                        Island(k, parnum) = max(Island(k, parnum), lu(1));
                        Island(k, parnum) = min(Island(k, parnum), lu(2));
                    end
                end
            end
        end
        fitness = benchmark_func(Island, problem); % calculate fitness
        [fitness_sorted, order] = sort(fitness); % the population was ordered by fitness
        sort_population = Island(order, :);
        % Replace the worst with the previous generation's elites
        n = length(Population);
        for k = 1 : Keep
            sort_population(n - k + 1, :) = chromKeep(k, :);
            fitness_sorted(n - k + 1) =  fitnessKeep(k);
        end
        [fitness_sorted, order] = sort(fitness_sorted); % the population was ordered again
        sort_population = sort_population(order, :);
        
        for i = 1 : OPTIONS.popsize % % save parent population of sort_population
            parent_population(i, :) = Population(i).chrom;
        end
        
        for i = 1 : OPTIONS.popsize
            Population(i).chrom = sort_population(i, :);
            Population(i).fitness = fitness_sorted(i);
        end
        % Make sure the population does not have duplicates
        Population = ClearDups(Population, lu(2), lu(1));
        for i = 1 : OPTIONS.popsize
            sort_population(i, :) = Population(i).chrom;
        end
        for i = 1 : OPTIONS.popsize
            if FES == 10000 * 0.1 || mod(FES, 10000) == 0
                [kk, ll] = min(fitness_sorted);
                RunValue(run, h) = kk;
                Para(h, :) = sort_population(ll, :);
                h = h + 1;
                fprintf('Algorithm:%s problemIndex:%d Run:%d FES:%d Best:%g\n','CPD_BBO',problem,run,FES,kk);
            end
            FES = FES + 1;
            if TimeFlag == 0
                if min(fitness_sorted) <= TEV
                    TempFES = FES;
                    TimeFlag = 1;
                end
            end
        end
        Iteration = Iteration + 1;
    end
    [kk, ll] = min(fitness_sorted);
    gbest = sort_population(ll, :);
    t2 = clock;
    RunTime(run) = etime(t2, t1);
    RunResult(run) = kk;
    RunFES(run) = TempFES;
    RunOptimization(run, 1 : OPTIONS.numVar) = gbest;
    RunParameter{run} = Para;
end
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Population] = GetSpeciesCounts(Population, P)
% Map fitness values to species counts
% This loop assumes the population is already sorted from most fit to least
% fit
for i = 1 : length(Population)
    if Population(i).fitness < inf
        Population(i).SpeciesCount = P - i;
    else
        Population(i).SpeciesCount = 0;
    end
end
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lambda, mu] = GetLambdaMu(Population, I, E, P)
% Calculate immigration rate and emigration rate for each species count
% lambda(i) is the immigration rate for individual i
% mu(i) is the emigration rate for individual i
for i = 1 : length(Population)
    lambda(i) = I * (1 - Population(i).SpeciesCount / P);
    mu(i) = E * Population(i).SpeciesCount / P;
end
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Population] = ClearDups(Population, MaxParValue, MinParValue)
% Make sure there are no duplicate individuals in the population.
% This logic does not make 100% sure that no duplicates exist, but any duplicates that are found are
% randomly mutated, so there should be a good chance that there are no duplicates after this procedure.
for i = 1 : length(Population)
    Chrom1 = sort(Population(i).chrom);
    for j = i+1 : length(Population)
        Chrom2 = sort(Population(j).chrom);
        if isequal(Chrom1, Chrom2)
            parnum = ceil(length(Population(j).chrom) * rand);
            Population(j).chrom(parnum) = floor(MinParValue + (MaxParValue - MinParValue + 1) * rand);
        end
    end
end
return;