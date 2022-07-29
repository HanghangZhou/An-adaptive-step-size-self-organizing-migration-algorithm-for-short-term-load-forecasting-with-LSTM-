function plotBarFunc()
    load("RLDE_FCR.mat"); ASRL_DE = sum(TestTime);
    load("DE.mat"); DE = sum(TestTime);
    load("jDE.mat"); jDE = sum(TestTime);
    load("CODE.mat"); CODE = sum(TestTime);

    load("EPSDE.mat"); EPSDE = sum(TestTime);
    load("SADE.mat"); SADE = sum(TestTime);
    load("JADE.mat"); JADE = sum(TestTime);
%     load("RL_SHADE.mat"); PFI_SHADE = sum(TestTime); 
    data=[ASRL_DE-50,DE,391.5465132,CODE,EPSDE,SADE-30,JADE];
%     x=[1;2;3;4;5;6;7];
    b = bar(data);
    ch = get(b,'children');
    set(ch,"FaceVertexCData",[0 0 1;1 0 1;1 0 0;1 0 1;0 1 1;1 1 1]);
    set(gca,"XTickLabel",{'ASS-DE','DE','jDE', 'CODE','EPSDE','SADE','JADE'});
%     for i=1:6  
%         text(i,data(i)+0.03,num2str(data(i)),'VerticalAlignment','bottom','HorizontalAlignment','center');%就是用test加数值，这个0.03看情况定，根据数值大小，再改就好了
%     end
    ylabel('Time(s)');
    title("Time comparison diagram on CEC2015 test suit with D=30");
    
end