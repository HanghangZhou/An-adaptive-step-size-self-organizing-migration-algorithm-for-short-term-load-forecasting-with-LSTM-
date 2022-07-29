function testLoad()
    persistent o;
    D=30;
    if isempty(o)
        load schwefel_102_data 
       if length(o)>=D
             o=o(1:D);
        else
             o=-100+200*rand(1,D);
        end
    %     initial_flag(2)=1;
    disp("yes");
    end
    disp("ok");
end