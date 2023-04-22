# Soluzioni domande aperte esercitazione 01

## Es 2

B) Domanda: la soluzione fornita è razionale rispetto alla misura
di performance? Se sì, motivare. Se no, modificarla
affinché lo sia

Risposta: Sì, è razionale in quanto raggiunge la cella verde nel minor numero
di step possibili. Nota: eliminando l'assunzione che la cella verde si trova nell'angolo
in basso a destra, non sarebbe stata possibile una soluzione
razionale per ogni possibile posizione della cella verde.

C) Domanda: esiste una soluzione razionale con un agente
reattivo semplice? Motivare la risposta

Risposta: No, non esiste una soluzione razionale con un agente reattivo
semplice perché, anche potendo osservare uno dei limiti della
griglia, non potrebbe "memorizzarlo" per poi cambiare direzione per
raggiungere il goal.

## Es 3

Domanda: Quale dei 4 tipi di agente è più adatto al caso in cui si
voglia passare per una cella intermedia prima di
raggiungere il goal?

Risposta: Il tipo più adatto è un agente goal-based. Questo perché un agente 
goal-based può mantenere in memoria, oltre alla propria percezione dello
stato dell'ambiente, anche informazioni sul proprio goal corrente e su
eventuali goal futuri.

## Es 4

B) Domanda: nel caso in cui non siano disponibili le azioni di
movimento diagonale, sarebbe comunque possibile
risolvere il task per tutte le possibili configurazioni
dell’ostacolo?

Risposta: Sì, sarebbe comunque possibile. Ad esempio, assumiamo che si usi la distanza
euclidea tra le coordinate della cella corrente e le coordinate della
cella verde come utility. In una configurazione senza ostacoli, un agente
che non può disporre delle azioni di movimento diagonali seguirebbe quindi
alternativamente azioni "E" ed "S", partendo da "E" 
(assumendo di risolvere i casi di ambiguità tra distanze di arrivo uguali
usando sempre l'azione "S"). Questo agente arriverebbe quindi al goal attraversando
le celle nell'ordine seguente: (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3),
e sarebbe razionale perché richiederebbe un numero minimo di step (7).
Si può dimostrare allora che aggiungendo l'ostacolo in una qualsiasi
posizione questo agente rimarrebbe comunque razionale. Prima di tutto,
si noti che se l'ostacolo non è lungo la traiettoria dell'agente
elencata sopra, nulla cambierebbe rispetto al caso senza ostacolo,
e l'agente sarebbe quindi ancora razionale. Se invece l'ostacolo
fosse una delle celle elencate sopra, possiamo distingure due casi,
a seconda che l'azione che l'agente esegue prima di trovarsi
di fronte all'ostacolo sia 1) "S" o 2) "E". Nel caso 1) l'ostacolo
si troverebbe a destra dell'agente; allora l'unica azione che
porterebbe a una maggiore utility sarebbe di nuovo "S", e quella
ancora successiva sarebbe "E". A questo punto l'agente si troverebbe
di nuovo sulla traiettoria originale senza aver compiuto step
aggiuntivi rispetto al caso senza ostacolo, e rimarrebbe quindi
razionale. Allo stesso modo, nel caso 2) l'agente invece di seguire
l'azione "S" che lo porterebbe a non muoversi, seguirebbe le azioni
"E" ed "S", ritornando di nuovo sulla traiettoria originale senza
step aggiuntivi.
