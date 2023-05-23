lesseq1(N1,N2) :- N1 =< N2.

% Knowledge base per l'esercizio 4

coauth(erdos,graham).
coauth(graham,erdos).

coauth(selberg,graham).
coauth(graham,selberg).

coauth(selberg,wickerson).
coauth(wickerson,selberg).

coauth(wickerson,erdos).
coauth(erdos,wickerson).
