s(){ printf "$1 w26 B w20 B w20 %.0s" $(seq $(($2+1))); }
BOOT="w200 START w60 START w60 START w300 A w200 A w60 A w60 A w60 A w900 w200 A w40 A w40 A w40 A w40 A w40 A w40 A w60"
pace(){ local o=""; for i in $(seq $1); do o="$o $(s UP 3) $(s DOWN 3)"; done; echo "$o"; }
AS(){ local o=""; for i in $(seq $1); do o="$o A w40 A w40 A w40 A w40 shot"; done; echo "$o"; }
TO_SOLUNE="$(s UP 12) $(s UP 8) $(s LEFT 10) $(s UP 14) $(s RIGHT 18) $(s LEFT 3) $(s UP 4) $(s LEFT 3) $(s UP 1) UP w30 shot"
TO_KIT="$(s UP 12) $(s UP 8) $(s LEFT 5) shot $(s UP 3) w60 shot"
RUNAWAY="w300 shot DOWN w20 RIGHT w20 A w200 shot B w60 B w60 shot"
# Battle mashing that picks the 4th move slot (bottom-right, e.g. Fury Cutter) every turn.
FC(){ local o=""; for i in $(seq $1); do o="$o A w40 DOWN w10 RIGHT w10 A w40 A w40 shot"; done; echo "$o"; }
# Safer move-4 picker: clear turn text with B (never selects), then Battle -> slot 4.
M4(){ local o=""; for i in $(seq $1); do o="$o B w60 B w60 B w60 B w60 B w60 B w60 B w60 B w60 shot A w40 DOWN w15 RIGHT w15 A w40"; done; echo "$o"; }
