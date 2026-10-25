#!/bin/bash
# Full-loop story flow (needs ECHO_STEPS_PER_PHASE 60). Prints runner script to stdout.
s(){ printf "$1 w26 B w20 B w20 %.0s" $(seq $(($2+1))); }
B6="B w30 B w30 B w30 B w30 B w30 B w30"
BOOT="w200 START w60 START w60 START w300 A w200 A w60 A w60 A w60 A w900 w200 A w40 A w40 A w40 A w40 A w40 A w40 A w60"
PRE=""; for i in $(seq 41); do PRE="$PRE $(s UP 3) $(s DOWN 3)"; done
WAIT=""; for i in $(seq 50); do WAIT="$WAIT $(s UP 3) $(s DOWN 3)"; done
A12=""; for i in $(seq 12); do A12="$A12 A w40"; done
echo "$BOOT $PRE $(s DOWN 9) $(s DOWN 15) $(s LEFT 8) w30 A w90 shot A w40 A w40 A w40 A w60 B w30 B w30 \
$(s RIGHT 6) $(s UP 26) $(s LEFT 4) $(s UP 2) w120 $(s UP 2) $(s RIGHT 3) $(s UP 1) A w60 $A12 shot $B6 \
$(s DOWN 1) $(s LEFT 3) $(s DOWN 3) w120 $(s RIGHT 4) $WAIT $B6 $(s LEFT 4) $(s DOWN 2) w120 save:/home/ubuntu/rom/st_house.ss \
$(s LEFT 1) DOWN w30 DOWN w120 START w40 shot $B6 save:/home/ubuntu/rom/st_out2.ss"
# Stage 2: walk to the Gloam Wood shrine and give Odette's farewell.
A=""; for i in $(seq 16); do A="$A A w40 A w40 A w40 A w40 A w40 shot"; done
echo "$(s RIGHT 4) $(s UP 10) $(s UP 8) $(s LEFT 10) $(s UP 14) $(s RIGHT 18) $(s LEFT 3) $(s UP 4) $(s LEFT 5) UP w30 shot save:/home/ubuntu/rom/st_shrine.ss A w60 shot $A w200 shot"
