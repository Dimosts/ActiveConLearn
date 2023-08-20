This is the repository contains the code for the active constraint acquisition systems described in our CP2023 paper "Guided Bottom-up Interactive Constraint Acquisition".

You can cite as follows: "Tsouros, D. C., Berden, S., & Guns, T. (2023). Guided Bottom-up Interactive Constraint Acquisition. In: 29th International Conference on Principles and Practice of Constraint Programming, CP 2023, Toronto, Canada, 2023


@inproceedings{tsouros2023guided,
  title={Guided Bottom-up Interactive Constraint Acquisition},
  author={Tsouros, Dimosthenis C and Berden, Senne and Guns, Tias},
  booktitle={29th International Conference on Principles and Practice of Constraint Programming, CP 2023, Toronto, Canada},
  year={2023},
}


## Usage 

main.py [-h] -a {quacq,mquacq,mquacq2,mquacq2-a,growacq}
               [-ia {quacq,mquacq,mquacq2,mquacq2-a}]
               [-qg {baseline,base,tqgen,pqgen}]
               [-obj {max,sol,p,prob,proba,class}] [-fs {1,2,3}] [-fc {1,2}]
               [-gqg] [-gfs] [-gfc]
               [-c {counts,random_forest,MLP,GaussianNB,CategoricalNB,SVM}]
               [-hls HIDDEN_LAYERS [HIDDEN_LAYERS ...]] [-lr LEARNING_RATE]
               [-t TIME_LIMIT] -b
               {9sudoku,4sudoku,jsudoku,random122,random495,new_random,golomb8,murder,job_shop_scheduling,exam_timetabling,exam_timetabling_simple,exam_timetabling_adv,exam_timetabling_advanced,nurse_rostering,nurse_rostering_simple,nurse_rostering_advanced,nurse_rostering_adv}
               [-nj NUM_JOBS] [-nm NUM_MACHINES] [-hor HORIZON] [-s SEED]
               [-nspd NUM_SHIFTS_PER_DAY] [-ndfs NUM_DAYS_FOR_SCHEDULE]
               [-nn NUM_NURSES] [-nps NURSES_PER_SHIFT] [-ns NUM_SEMESTERS]
               [-ncps NUM_COURSES_PER_SEMESTER] [-nr NUM_ROOMS]
               [-ntpd NUM_TIMESLOTS_PER_DAY] [-ndfe NUM_DAYS_FOR_EXAMS]
               [-np NUM_PROFESSORS]

## Options:

  -h, --help            show this help message and exit

  -a {quacq,mquacq,mquacq2,mquacq2-a,growacq}, --algorithm {quacq,mquacq,mquacq2,mquacq2-a,growacq}
                        The name of the algorithm to use

  -ia {quacq,mquacq,mquacq2,mquacq2-a}, --inner-algorithm {quacq,mquacq,mquacq2,mquacq2-a}
                        Only relevant when the chosen algorithm is GrowAcq -
                        the name of the inner algorithm to use

  -qg {baseline,base,tqgen,pqgen}, --query-generation {baseline,base,tqgen,pqgen}
                        The version of the query generation method to use

  -obj {max,sol,p,prob,proba,class}, --objective {max,sol,p,prob,proba,class}
                        The objective function used in query generation

  -fs {1,2,3}, --findscope {1,2,3}
                        The version of the findscope method to use

  -fc {1,2}, --findc {1,2}
                        The version of the findc method to use

  -gqg, --guide-qgen    Use this to guide query generation

  -gfs, --guide-findscope   Use this to guide FindScope

  -gfc, --guide-findc   Use this to guide FindC

  -c {counts,random_forest,MLP,GaussianNB,CategoricalNB,SVM}, --classifier {counts,random_forest,MLP,GaussianNB,CategoricalNB,SVM}
                        Only relevant when the chosen query generation method
                        is 4 - the machine learning classifier to use

  -hls HIDDEN_LAYERS [HIDDEN_LAYERS ...], --hidden-layers HIDDEN_LAYERS [HIDDEN_LAYERS ...]
                        Only relevant when the chosen query generation method
                        is 4 and the chosen classifier is MLP - the hidden
                        layer shapes, e.g. -hls 16 32 64 creates 3 hidden
                        layers of sizes 16, 32 and 64, respectively

  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Only relevant when the chosen query generation method
                        is 4 and the chosen classifier is MLP - the learning
                        rate

  -t TIME_LIMIT, --time-limit TIME_LIMIT
                        An optional time limit

  -b {9sudoku,4sudoku,jsudoku,random122,random495,new_random,golomb8,murder,job_shop_scheduling,exam_timetabling,exam_timetabling_simple,exam_timetabling_adv,exam_timetabling_advanced,nurse_rostering,nurse_rostering_simple,nurse_rostering_advanced,nurse_rostering_adv}, --benchmark {9sudoku,4sudoku,jsudoku,random122,random495,new_random,golomb8,murder,job_shop_scheduling,exam_timetabling,exam_timetabling_simple,exam_timetabling_adv,exam_timetabling_advanced,nurse_rostering,nurse_rostering_simple,nurse_rostering_advanced,nurse_rostering_adv}
                        The name of the benchmark to use

  -nj NUM_JOBS, --num-jobs NUM_JOBS
                        Only relevant when the chosen benchmark is job-shop
                        scheduling - the number of jobs

  -nm NUM_MACHINES, --num-machines NUM_MACHINES
                        Only relevant when the chosen benchmark is job-shop
                        scheduling - the number of machines

  -hor HORIZON, --horizon HORIZON
                        Only relevant when the chosen benchmark is job-shop
                        scheduling - the horizon

  -s SEED, --seed SEED  Only relevant when the chosen benchmark is job-shop
                        scheduling - the seed

  -nspd NUM_SHIFTS_PER_DAY, --num-shifts-per-day NUM_SHIFTS_PER_DAY
                        Only relevant when the chosen benchmark is nurse
                        rostering - the number of shifts per day

  -ndfs NUM_DAYS_FOR_SCHEDULE, --num-days-for-schedule NUM_DAYS_FOR_SCHEDULE
                        Only relevant when the chosen benchmark is nurse
                        rostering - the number of days for the schedule

  -nn NUM_NURSES, --num-nurses NUM_NURSES
                        Only relevant when the chosen benchmark is nurse
                        rostering - the number of nurses

  -nps NURSES_PER_SHIFT, --nurses-per-shift NURSES_PER_SHIFT
                        Only relevant when the chosen benchmark is nurse
                        rostering (advanced) - the number of nurses per shift

  -ns NUM_SEMESTERS, --num-semesters NUM_SEMESTERS
                        Only relevant when the chosen benchmark is exam
                        timetabling - the number of semesters

  -ncps NUM_COURSES_PER_SEMESTER, --num-courses-per-semester NUM_COURSES_PER_SEMESTER
                        Only relevant when the chosen benchmark is exam
                        timetabling - the number of courses per semester

  -nr NUM_ROOMS, --num-rooms NUM_ROOMS
                        Only relevant when the chosen benchmark is exam
                        timetabling - the number of rooms

  -ntpd NUM_TIMESLOTS_PER_DAY, --num-timeslots-per-day NUM_TIMESLOTS_PER_DAY
                        Only relevant when the chosen benchmark is exam
                        timetabling - the number of timeslots per day

  -ndfe NUM_DAYS_FOR_EXAMS, --num-days-for-exams NUM_DAYS_FOR_EXAMS
                        Only relevant when the chosen benchmark is exam
                        timetabling - the number of days for exams

  -np NUM_PROFESSORS, --num-professors NUM_PROFESSORS
                        Only relevant when the chosen benchmark is exam
                        timetabling - the number of professors