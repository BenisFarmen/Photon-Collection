#A script for automatic preprocessing with parameters on the cluster
import subprocess

source_folder = input("Type in the source folder:")
destination_folder = input("Type output folder:")

#copy files from local to remote
subprocess.run(["rsync -R" + " " + source_folder + " " + destination_folder])

#switch working directory to the right place
subprocess.run(["cd" + " " + destination_folder])

#Unzip data if it is not unzipped yet
subprocess.run(["gunzip *nii.gz"])

#make list to execute the job in the end
subprocess.run(["readlink -f *.nii > list.txt"])


#ask for all necessary script parameters
#How much memory is needed (per node). Possible units: K, G, M, T
memory_needed = input("Type the amount of memory needed per node (Possible Units are K, G, M T):")
#Specify the number of jobs
number_of_jobs = input("Type the number of jobs you need:")
#set max wallclock time
max_wallclock_time = input("Set maximum wallclock time (hh:mm:ss):")
#set name of job
job_name = input("Set name of job:")
#send mail to this address
email_adress = input("Please type your e-mail adress:")
#preprocessing directory gets set
prepro_directory = "/sratch/tmp/trap/preprocessing/directory"

f= open("submit.cmd","w+")

f.write("#!bin/bash\n#SBATCH --mem=" + memory_needed + "\n"
        + "#SBATCH --array=1-" + number_of_jobs + "%40" + "\n"
        + "#SBATCH --partition normal,requeue\n"
        + "#SBATCH --time=" + max_wallclock_time + "\n"
        + "#SBATCH --job-name=" + job_name + "\n"
        + "#SBATCH --mail-type=ALL\n"
        + "#SBATCH --output output.dat\n"
        + "#SBATCH --m$ail-user=" + email_adress +"\n"
        + "cd " + prepro_directory + "\n"
        + "cat list.txt | head -n $SLURM_ARRAY_TASK_ID | tail -n 1 > $arg2$/list_${SLURM_ARRAY_TASK_ID}.txt\n"
        + "spm12/run_spm12.sh mcr/v94/ cat12 $arg2$/list_${SLURM_ARRAY_TASK_ID}.txt")

f.close()

subprocess.run(["sbatch submit.cmd"])
