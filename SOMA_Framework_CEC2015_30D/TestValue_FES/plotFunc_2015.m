function  plotFunc_2015()
% %r  红    g    绿 b     蓝 c   蓝绿
% %m  紫红   y  黄  k   黑   w   白
% %v?向下三角形   ^?向上三角形   *?星号   p?五角星   +?加号h?六角星??
% % +?加号h?六角星 o 圆s 正方形 
    load("ASOMA.mat"); 
    RLDE_strategy = TestValue;
    load("PSO.mat"); 
    DE = TestValue;
    load("CPSO.mat"); 
    jDE = TestValue;
    load("SOMA.mat"); 
    CODE = TestValue;
    load("LSOMA.mat"); 
    EPSDE = TestValue;
    load("OSOMA.mat"); 
    SADE = TestValue;
%     load("JADE.mat"); 
%     JADE = TestValue;
%     load("RLDE.mat"); 
%     RLDE = TestValue;
%     load("qIDE.mat"); 
%     qIDE = TestValue;
    
    for i = 15:-1:1
        figure(i);
        
        tmpRLDE_strategy = RLDE_strategy{i,1};
        tmpDE = DE{i,1};
        tmpjDE = jDE{i,1};
        tmpACODE = CODE{i,1};
        tmpEPSDE = EPSDE{i,1};
        tmpSADE = SADE{i,1};
%         tmpJADE = JADE{i,1};
%         tmpRLDE = RLDE{i,1};
%         tmpqIDE = qIDE{i,1};
        
        [~,py] = size(tmpRLDE_strategy);
        j = 1:py;
        plot(j,log(tmpRLDE_strategy),"r-o"); hold on;  grid on;
        plot(j,log(tmpDE),"y-+"); hold on;  grid on;
        plot(j,log(tmpjDE),"m-*"); hold on;  grid on;
        plot(j,log(tmpACODE),"c-s"); hold on;  grid on;
        plot(j,log(tmpEPSDE),"g-d"); hold on;  grid on;
        plot(j,log(tmpSADE),"b-<"); hold on;  grid on;
%         plot(j,log(tmpJADE),"k-p"); hold on;  grid on; 
%         plot(j,log(tmpRLDE),"m-o"); hold on;  grid on;
%         plot(j,log(tmpqIDE),"b-d"); hold on;  grid on; 
        

        xlabel("Sampling point");
        ylabel("log(fitness)");
        titleName = sprintf("F%s",num2str(i));
        title(titleName);
        legend("SSOMA","PSO","CPSO","SOMA","LSOMA","OSOMA");
        box on;
    end
end

