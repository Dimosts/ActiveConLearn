rem Reproducing our CP2023 experiments

rem Q1] Table 2

rem Baseline

	python main.py -a mquacq2-a -b jsudoku -qg baseline

	python main.py -a mquacq2-a -b random495 -qg baseline

	python main.py -a mquacq2-a -b murder -qg baseline

	python main.py -a mquacq2-a -b golomb8 -qg baseline

	python main.py -a mquacq2-a -b "job_shop_scheduling" -nj "10" -nm "3" -hor "15" -s "0" -qg baseline

rem TQ-Gen

	python main.py -a mquacq2-a -b jsudoku -qg tqgen -t 2

	python main.py -a mquacq2-a -b random495 -qg tqgen -t 2

	python main.py -a mquacq2-a -b murder -qg tqgen -t 2

	python main.py -a mquacq2-a -b golomb8 -qg tqgen -t 2

	python main.py -a mquacq2-a -b "job_shop_scheduling" -nj "10" -nm "3" -hor "15" -s "0" -qg tqgen -t 2

rem PQ-Gen

	python main.py -a mquacq2-a -b jsudoku -qg pqgen

	python main.py -a mquacq2-a -b random495 -qg pqgen

	python main.py -a mquacq2-a -b murder -qg pqgen

	python main.py -a mquacq2-a -b golomb8 -qg pqgen

	python main.py -a mquacq2-a -b "job_shop_scheduling" -nj "10" -nm "3" -hor "15" -s "0" -qg pqgen


rem [Q2] Table 3: GrowAcq + MQuAcq-2

	python main.py -a growacq -ia mquacq2-a -b jsudoku -qg pqgen

	python main.py -a growacq -ia mquacq2-a -b random495 -qg pqgen

	python main.py -a growacq -ia mquacq2-a -b murder -qg pqgen

	python main.py -a growacq -ia mquacq2-a -b golomb8 -qg pqgen

	python main.py -a growacq -ia mquacq2-a -b "job_shop_scheduling" -nj "10" -nm "3" -hor "15" -s "0" -qg pqgen


rem [Q3 - Q4] Table 3: GrowAcq + MQuAcq-2 guided

	python main.py -a growacq -ia mquacq2-a -b jsudoku -qg pqgen -obj proba

	python main.py -a growacq -ia mquacq2-a -b random495 -qg pqgen -obj proba

	python main.py -a growacq -ia mquacq2-a -b murder -qg pqgen -obj proba

	python main.py -a growacq -ia mquacq2-a -b golomb8 -qg pqgen -obj proba

	python main.py -a growacq -ia mquacq2-a -b "job_shop_scheduling" -nj "10" -nm "3" -hor "15" -s "0" -qg pqgen -obj proba


rem [Q5] Table 4: GrowAcq + MQuAcq-2 guided

rem	python main.py -a growacq -ia mquacq2-a -b "job_shop_scheduling" -nj "15" -nm "11" -hor "40" -s "0" -qg pqgen -obj proba
	
rem	python main.py -a growacq -ia mquacq2-a -b "job_shop_scheduling" -nj "19" -nm "12" -hor "40" -s "0" -qg pqgen -obj proba
