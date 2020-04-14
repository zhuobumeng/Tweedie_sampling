import os
import argparse


def run(args):
    for i in range(args.repeat):
        random_seed = args.random_seed + i
        args.jobname = args.jobid = "td" + str(args.random_seed)
        filename = "tweedie_n" + str(args.n) + "_seed" + \
            str(random_seed) + ".sh"
        with open(filename, "w") as file:
            file.write(f'''#!/bin/bash

#SBATCH --job-name={args.jobname}
#SBATCH --output={args.jobid}.out
#SBATCH --error={args.jobid}.err
#SBATCH --nodes=1
#SBATCH --partition=broadwl
#SBATCH --ntasks=1

module load cuda/10.0

python train_tweedie.py \\
--n {args.n} \\
--s {args.s} \\
--lam {args.lam} \\
--num_iter {args.num_iter} \\
--random_seed {random_seed}
''')

        os.system("chmod +x " + filename)
        print("sbatch " + filename)
        os.system("sbatch " + filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basics
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--s', type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument('--num_iter', type=int, default=8000)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument("--mtd", type=str, default="horseshoe")
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--lse_init", type=int, default=1)

    args = parser.parse_args()
    run(args)
