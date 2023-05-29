
lesseq1(N1,N2) :- N1 =< N2.

% se X <= Y, il minimo è X
minimum(X,Y,X) :- lesseq1(X,Y).
% se Y <= X, il minimo è Y
minimum(X,Y,Y) :- lesseq1(Y,X).

% B^0 = 1
pow1(_,0,1).
% B^1 = B
pow1(B,1,B).
% B^E = B * B^(E-1)
pow1(B,E,Z) :-
  E1 is E-1,
  pow1(B,E1,Z1),
  Z is Z1*B.

% sum(N) = N + sum(N-1)
sum(0,0).
sum(1,1).
sum(N,Z) :- N1 is N-1, sum(N1,Z1), Z is Z1+N.

% [] è un suffisso di qualsiasi lista
suffix([],_).
% ogni lista è suffisso di se stessa
suffix(L,L).
% se una lista S è il suffisso di L, allora è anche il suffisso di
% cons(X,L) per qualsiasi X
% Possibile sol alternativa: usare prefix e reverse
suffix(S,[_|L]) :- suffix(S,L). 

% L'insieme vuoto è sottoinsieme di qualsiasi insieme.
subset([],_).
% Se A è sottoinsieme di B e X è in B, allora anche ({X} U A) è un
% sottoinsieme di B. 
subset([X|A],B) :- member(X,B), subset(A,B). 

% L'intersezione con l'insieme vuoto è l'insieme vuoto.
intersection([],_,[]).
intersection(_,[],[]).
% Se X è in B e C è l'intersezione tra A e B, allora ({X} U C) è
% l'intersezione tra (A U {X}) e B.
intersection([X|A],B,[X|C]) :- member(X,B), intersection(A,B,C).
% Se C è l'intersezione tra A e B e X non è in B, allora l'intersezione
% tra ({X} U A) e B rimane C.
intersection([_|A],B,C) :- intersection(A,B,C).


% Knowledge base per l'esercizio 4

coauth(erdos,graham).
coauth(graham,erdos).

coauth(selberg,graham).
coauth(graham,selberg).

coauth(selberg,wickerson).
coauth(wickerson,selberg).

coauth(wickerson,erdos).
coauth(erdos,wickerson).

% Struttura della soluzione:
% - candidate calcola un possibile percorso di autori che arriva ad erdos
% (e lunghezza associata)
% - mincandidate calcola il candidato a lunghezza minore
% - erdosnum usa mincandidate per calcolare il numero di erdos

% Per arrivare a erdos partendo da erdos, dato che il percorso già
% seguito è Path, la soluzione è [erdos|Path]. Il costo residuo
% (i.e. non considerando la lunghezza di Path), è 1.
candidate(Path, erdos, [erdos|Path], 1).
% Se:
% 1. Node1 è coautore di Node
% 2. Node1 non è già nel percorso corrente (per evitare cicli)
% 3. C'è un percorso Sol per arrivare ad erdos dal primo nodo di Path,
%    passando per Node1, di costo residuo PrevCost
% -> allora Sol è un percorso dal primo nodo di Path che passa per
%    Node e ha costo residuo PrevCost+1.
% Nota: se Path è [], allora Cost coincide con la lunghezza di Sol.
candidate(Path, Node, Sol, Cost)  :-
  coauth(Node, Node1),
  \+ member(Node1, Path), % evita cicli
  candidate([Node | Path], Node1, Sol, PrevCost),
  Cost is PrevCost+1.

% Il percorso minimo per arrivare ad erdos, ovvero quello tale che non
% ne esistono altri con costo minore.
mincandidate(X, MinCost, Path) :-
    candidate([], X, Path, MinCost),
    \+ (candidate([], X, OtherPath, LowerCost),
        OtherPath \= Path,
        LowerCost < MinCost).

erdosnum(X,N) :- mincandidate(X, C, _), N is C-1.
