function All_Run()
clc
clear all
close all
format long
format compact
addpath('Public');
addpath('SOMAs');
addpath('CEC2015');%Fun_Num=25;
TEV=Error(Dim(1));
runmax=25; 
Fun_Num=15;

%算法内部参数
N=100;               %population size

% 1：SOMA, 2:OSOMA, 3:ASOMA, 4:PSO, 5:CPSO
% 6: LSOMA,7:HBSOMA
% 8: SPIRAL 9: ASPIRAL
% 10:JIA
for algorithm=3:3 %选择算法接口
    Datime = date;
    TestFitness=[];
    TestResult=[];
    TestValue={};
    TestTime=[];
    TestRatio=[];
    TestFES=[];
    TestOptimization={};
    TestParameter={};
    switch algorithm   
        %% SOMA
        case 1
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=SOMA(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','leadr_test');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','leadr_test');
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','leadr_test');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','leadr_test');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','leadr_test');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','leadr_test');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','leadr_test');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','leadr_test');
            save(Test,'TestParameter');
            
        %% OSOMA
        case 2
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=OSOMA(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','OSOMA');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','OSOMA');
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','OSOMA');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','OSOMA');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','OSOMA');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','OSOMA');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','OSOMA');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','OSOMA');
            save(Test,'TestParameter');
            
        %% ASSOMA
        case 3
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=ASSOMA(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','ESOMA');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','ESOMA');
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','ESOMA');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','ESOMA');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','ESOMA');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','ESOMA');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','ESOMA');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','ESOMA');
            save(Test,'TestParameter');

        %% PSO
        case 4
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=PSO(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','PSO');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','PSO');
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','PSO');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','PSO');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','PSO');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','PSO');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','PSO');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','PSO');
            save(Test,'TestParameter');
%             
         %% CPSO   
        case 5
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=CPSO(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end 
            Test=sprintf('TestFitness/%s.mat','CPSO');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','CPSO');
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','CPSO');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','CPSO');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','CPSO');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','CPSO');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','CPSO');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','CPSO');
            save(Test,'TestParameter');
            
         %% LSOMA   
        case 6
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=LSOMA(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','LSOMA');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','LSOMA');
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','LSOMA');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','LSOMA');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','LSOMA');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','LSOMA');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','LSOMA');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','LSOMA');
            save(Test,'TestParameter');
            
         %% HBSOMA   
        case 7
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=HBSOMA(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','HBSOMA');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','HBSOMA');
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','HBSOMA');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','HBSOMA');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','HBSOMA');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','HBSOMA');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','HBSOMA');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','HBSOMA');
            save(Test,'TestParameter');
            
         %% SPIRAL
         case 8
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=SPIRAL(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','SPIRAL');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','SPIRAL');
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','SPIRAL');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','SPIRAL');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','SPIRAL');
            save(Test,'TestRatio'); 
            Test=sprintf('TestFES/%s.mat','SPIRAL');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','SPIRAL');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','SPIRAL');
            save(Test,'TestParameter');
            
         %% ASPIRAL
         case 9
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=ASPIRAL(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','ASPIRAL_s0.42');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','ASPIRAL_s0.42'); 
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','ASPIRAL_s0.42');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','ASPIRAL_s0.42');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','ASPIRAL_s0.42');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','ASPIRAL_s0.42');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','ASPIRAL_s0.42');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','ASPIRAL_s0.42');
            save(Test,'TestParameter');
            
            %% JIA
         case 10
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=JIA(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','JIA');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','JIA'); 
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','JIA');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','JIA');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','JIA');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','JIA');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','JIA');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','JIA');
            save(Test,'TestParameter');
            
           %% MAGA
         case 11
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=MAGA(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','MAGA');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','MAGA'); 
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','MAGA');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','MAGA');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','MAGA');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','MAGA');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','MAGA');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','MAGA');
            save(Test,'TestParameter');
            
            %% RLDE_strategy
         case 12
            for problem=1:15
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=RLDE_strategy2(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','2RLDE_strategy');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','2RLDE_strategy'); 
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','2RLDE_strategy');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','2RLDE_strategy');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','2RLDE_strategy');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','2RLDE_strategy');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','2RLDE_strategy');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','2RLDE_strategy');
            save(Test,'TestParameter');
            
             %% RLDE_strategy3
         case 13
            for problem=1:Fun_Num
                [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=RLDE_strategy3(problem,N,runmax);
                sign=(RunResult<=TEV(problem));
                Ratio=sum(sign)/runmax;
                FES=sign.*RunFES;
                TestFitness=[TestFitness;RunResult];
                TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
                TestValue=[TestValue;mean(RunValue)];
                TestTime=[TestTime;mean(RunTime)];
                TestRatio=[TestRatio;Ratio];
                TestFES=[TestFES;mean(FES)];
                TestOptimization=[TestOptimization;RunOptimization];
                TestParameter=[TestParameter;RunParameter];
            end
            Test=sprintf('TestFitness/%s.mat','RLDE_strategy3');
            save(Test,'TestFitness');
            Test=sprintf('TestResult/%s.mat','RLDE_strategy3'); 
            save(Test,'TestResult');
            Test=sprintf('TestValue_FES/%s.mat','RLDE_strategy3');
            save(Test,'TestValue');
            Test=sprintf('TestTime/%s.mat','RLDE_strategy3');
            save(Test,'TestTime');
            Test=sprintf('TestRatio/%s.mat','RLDE_strategy3');
            save(Test,'TestRatio');
            Test=sprintf('TestFES/%s.mat','RLDE_strategy3');
            save(Test,'TestFES');
            Test=sprintf('TestOptimization/%s.mat','RLDE_strategy3');
            save(Test,'TestOptimization');
            Test=sprintf('TestParameter_FES/%s.mat','RLDE_strategy3');
            save(Test,'TestParameter');
    end
    
end

