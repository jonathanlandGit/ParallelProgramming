-module(erlang2).
-compile(export_all).
-export([reduce/2]).

reduce(Func, List) ->
    reduce(root, Func, List).

%When done send results to Parent
reduce(Parent, _, [A]) -> 
	%send to parent 
    Parent ! { self(), A}; 
    	 
%get contents of list, apply function and store in Parent
reduce(Parent, Func, List) ->
            { Left, Right } = lists:split(trunc(length(List)/2), List),
            Me = self(),
            %io:format("Splitting in two~n"),
            Pl = spawn(fun() -> reduce(Me, Func, Left) end),
            Pr = spawn(fun() -> reduce(Me, Func, Right) end),
            %merge results in parent and call Func on final left and right halves
            combine(Parent, Func,[Pl, Pr]).

%merge pl and pl and combine in parent
combine(Parent, Func, [Pl, Pr]) ->
    %wait for processes to complete (using receive) and then send to Parent
    receive
        { Pl, Sorted } -> combine(Parent, Func, Pr, Sorted);
        { Pr, Sorted } -> combine(Parent, Func, Pl, Sorted)
    end.
        
combine(Parent, Func, P, List) ->
    %wait and store in results and then call ! to send
    receive
        { P, Sorted } ->
            Results = Func(Sorted, List),
            case Parent of
                root ->
                    Results;
                %send results to parent
                _ -> Parent ! {self(), Results}
            end
    end.


