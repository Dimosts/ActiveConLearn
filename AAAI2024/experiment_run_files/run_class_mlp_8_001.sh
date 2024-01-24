#!/bin/bash

# run gowacq + mquacq2 + a + MLP -hls 8 -lr 0.01
for i in {1..10}
do
	python3.8 main.py -a growacq -ia mquacq2-a -fs 2 -fc 1 -b nurse_rostering_adv -nspd 3 -ndfs 7 -nn 18 -nps 5 -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2-a -fs 2 -fc 1 -b 9sudoku -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2-a -fs 2 -fc 1 -b jsudoku -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2-a -fs 2 -fc 1 -b new_random -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2-a -fs 2 -fc 1 -b "job_shop_scheduling" -nj "10" -nm "3" -hor "15" -s "0" -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2-a -fs 2 -fc 1 -b "exam_timetabling" -ns 8 -ncps 6 -nr 3 -ntpd 3 -ndfe 10 -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01		
done

# run gowacq + mquacq2 + MLP -hls 8 -lr 0.01
for i in {1..10}
do
	python3.8 main.py -a growacq -ia mquacq2 -fs 2 -fc 1 -b nurse_rostering_adv -nspd 3 -ndfs 7 -nn 18 -nps 5 -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2 -fs 2 -fc 1 -b 9sudoku -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2 -fs 2 -fc 1 -b jsudoku -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2 -fs 2 -fc 1 -b new_random -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2 -fs 2 -fc 1 -b "job_shop_scheduling" -nj "10" -nm "3" -hor "15" -s "0" -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01
	python3.8 main.py -a growacq -ia mquacq2 -fs 2 -fc 1 -b "exam_timetabling" -ns 8 -ncps 6 -nr 3 -ntpd 3 -ndfe 10 -qg pqgen -gqg -o class -c MLP -hls 8 -lr 0.01		
done
