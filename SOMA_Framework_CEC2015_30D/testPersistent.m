f1();
f1();
f2();
f2();

function f1()
    persistent a;
    if isempty(a)
        a = 0;
    end
    a = a + 1
end
function f2()
    persistent a;
    if isempty(a)
        a = 10;
    end
    a = a - 2
end
