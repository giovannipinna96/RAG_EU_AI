# Demetra HPC — Command-Line User Guide

> University of Trieste — [https://demetra.units.it/docs/](https://demetra.units.it/docs/)

---

## 1. Overview

Demetra is the HPC cluster of the University of Trieste. It is a hybrid environment where compute-intensive workloads run on bare-metal servers, while auxiliary services are virtualized on a VMware hypervisor. All nodes run **Rocky Linux 9.3**. Job scheduling is handled by **SLURM** with cgroup support for resource isolation.

**In numbers:**

| Totals | Value |
|---|---|
| Physical servers | 11 |
| CPU cores | 476 |
| RAM | ~5.8 TB |
| Local storage | ~35 TB |
| Shared storage (NetApp) | ~374 TB |

---

## 2. Infrastructure at a Glance

### 2.1 Login / Management Node

The single entry point to the cluster is the **demetra** node. It serves as both the shared login node and the SLURM controller.

| | |
|---|---|
| Hostname | `demetra.units.it` |
| IP | `140.105.164.221` |
| Role | Login node + SLURM controller |

> **Code of conduct:** Do not run compute-heavy or long-running tasks on the login node. Resource limits are enforced. Use SLURM to send work to the compute nodes.

### 2.2 Compute Nodes

| Node(s) | CPU | RAM | GPU |
|---|---|---|---|
| **babbage** (Dell R740) | 2× Intel Xeon Gold 6140 (18c) @ 2.30 GHz | 768 GB | 1× NVIDIA Tesla V100 32 GB, 1× NVIDIA A100 40 GB |
| **lovelace-01, -02** (Dell R7525) | 2× AMD EPYC 7542 (32c) | 768 GB | 2× NVIDIA A100 each (one full + one split via MIG) |
| **turing-01** (HPE DL560 Gen10) | 4× Intel Xeon Gold 6140 (18c) @ 2.30 GHz | 1 TB | — |
| **turing-02, -03** (HPE DL560 Gen10) | 4× Intel Xeon Gold 6140 (18c) @ 2.30 GHz | 1 TB | — |
| **nodeppc01** (IBM Power System) | 2× IBM POWER9 (32c total) | 320 GB | 4× NVIDIA Tesla V100 |
| **nodeppc02, -04, -05** (IBM Power System) | 2× IBM POWER9 (32c total) | 256 GB | 4× NVIDIA Tesla V100 each |
| **nodeppc03** (IBM Power System) | 2× IBM POWER9 (32c total) | 256 GB | 2× NVIDIA Tesla V100 |

### 2.3 Storage

The shared storage is a **NetApp FAS2720** (2× DS212C + 1× DS460C enclosures, ~374 TB total). The following volumes are exported to all nodes via NFSv4:

| Mount point | Purpose | Size |
|---|---|---|
| `/u` | User home directories | 2 TB |
| `/share` | Group shared data | 24 TB |
| `/opt` | Shared software (separate export on PPC nodes) | 200 GB |

Your personal paths are:

```
/u/<username>                   # home directory
/share/<group>/<username>       # shared data space
```

---

## 3. Getting Access

### 3.1 Request an Account

Send an email to:

- **To:** `support+migemigra@exact-lab.it`
- **Cc:** `mige.hpc@units.it`

Include the following information: your university ID, preferred username (optional — your numeric ID is used otherwise), your research group, and your supervisor's name.

### 3.2 Connect via SSH

Demetra is accessible from the **UniTS internal network** or through the **UniTS VPN**.

```bash
ssh <your-university-ID>@demetra.units.it
```

Authentication uses university credentials (ID and password). Your preferred username is an alias — use your university ID to log in.

### 3.3 Tips for VS Code Users

Instead of running VS Code Server on the login node (which is not allowed), consider these alternatives:

1. **`sshfs` VS Code plugin** — runs VS Code locally while interacting with the remote filesystem and terminal.
2. **`sshfs` mount on Linux** — mount the remote filesystem as a network drive.
3. **`rsync`** — synchronize a remote directory with a local one (only transfers changes).
4. **`rclone`** — supports sync and mounting of remote SSH servers (works on non-Linux too).

---

## 4. SLURM — The Queue System

All compute work must go through SLURM. The login node is **not** for computation.

### 4.1 Essential Commands

| Command | Usage | Description |
|---|---|---|
| `sbatch` | `sbatch script.sh` | Submit a batch job script |
| `srun` | `srun <options> <command>` | Run a job interactively |
| `squeue` | `squeue -u $USER` | Show your queued/running jobs |
| `scancel` | `scancel <job-id>` | Cancel a job |
| `sacct` | `sacct` | Show info on current and past jobs |
| `sinfo` | `sinfo` | Show partition and node status |

### 4.2 Partitions

Four partitions are configured. Request one (or more) with `#SBATCH --partition <name>`. Specifying multiple partitions increases your chances of getting scheduled sooner.

| Partition | Nodes | Max Walltime | Resource Limits | Use Case |
|---|---|---|---|---|
| `Main` | babbage | default | default | General + GPU jobs |
| `lovelace` | lovelace-01, -02 | default | default | GPU-intensive work |
| `turing-long` | turing-01, -02, -03 | 2 weeks | max 8 CPUs, max 64 GB RAM | Long-running "thin" jobs |
| `turing-wide` | turing-01, -02, -03 | 2 days | no limits | Short-running "wide" jobs |

Check current partition status:

```bash
sinfo
```

### 4.3 Writing a SLURM Batch Script

A SLURM batch script is a regular shell script with `#SBATCH` directives. Here is a basic template:

```bash
#!/bin/bash
#SBATCH --job-name=my_job          # job name shown in squeue
#SBATCH --partition=Main           # partition to submit to
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks=1                 # number of tasks (MPI ranks)
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem=16G                  # total memory
#SBATCH --time=02:00:00            # max walltime (HH:MM:SS)
#SBATCH --output=job_%j.out        # stdout file (%j = job ID)
#SBATCH --error=job_%j.err         # stderr file

# Load any modules you need
module load <software>

# Run your program
./my_program
```

Submit it:

```bash
sbatch script.sh
```

### 4.4 Common SBATCH Directives Reference

```bash
#SBATCH --job-name=NAME            # human-readable job name
#SBATCH --partition=PART           # partition (Main, lovelace, turing-long, turing-wide)
#SBATCH --nodes=N                  # number of nodes
#SBATCH --ntasks=N                 # total MPI tasks
#SBATCH --ntasks-per-node=N        # MPI tasks per node
#SBATCH --cpus-per-task=N          # cores per task (for OpenMP / multithreading)
#SBATCH --mem=SIZE                 # memory per node (e.g. 32G)
#SBATCH --time=DD-HH:MM:SS        # maximum walltime
#SBATCH --output=FILE              # stdout (%j = job ID, %x = job name)
#SBATCH --error=FILE               # stderr
#SBATCH --gres=gpu:TYPE:N          # GPU request (see Section 5)
#SBATCH --mail-type=END,FAIL       # email notifications
#SBATCH --mail-user=you@units.it   # your email
```

### 4.5 Interactive Sessions

To get a shell on a compute node:

```bash
srun -n1 --pty bash
```

Request more resources for interactive work:

```bash
srun --partition=Main --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
```

**Note for PPC nodes:** use the `-l` flag to re-initialize environment variables, since those nodes have a separate modules system:

```bash
srun -p ppc -N 1 --pty /bin/bash -l
```

### 4.6 Monitoring Jobs

```bash
# View your jobs
squeue -u $USER

# View all jobs
squeue

# Detailed info on a running/completed job
sacct -j <job-id> --format=JobID,JobName,Partition,State,Elapsed,MaxRSS

# View job output in real-time
tail -f slurm-<job-id>.out

# Cancel a job
scancel <job-id>

# Cancel all your jobs
scancel -u $USER
```

---

## 5. GPU Jobs

Three GPU-equipped nodes are available on the x86 side: `babbage`, `lovelace-01`, `lovelace-02`.

### 5.1 Available GPUs

| Node | GPUs |
|---|---|
| babbage | 1× NVIDIA A100 40 GB + 1× NVIDIA V100 32 GB |
| lovelace-01 | 1× full A100 + 1× A100 split via MIG (3 virtual GPUs) |
| lovelace-02 | 1× full A100 + 1× A100 split via MIG (3 virtual GPUs) |

The PPC nodes (nodeppc01–05) also have NVIDIA Tesla V100 GPUs (2–4 per node).

### 5.2 Requesting a Full GPU

```bash
# Interactive
srun -n1 --gres=gpu:1 --pty bash

# In a batch script
#SBATCH --gres=gpu:1
```

This allocates one full (physical) A100.

### 5.3 Requesting a MIG (Virtual GPU) Partition

If you only need a portion of a GPU, you can request MIG devices by profile name:

```bash
# Request 2 MIG slices of the 1g.20gb profile
srun -n1 --gres=gpu:1g.20gb:2 --pty bash
```

### 5.4 Querying Available GPU Devices

```bash
scontrol show node lovelace-01 | grep -i gres
```

Example output:

```
Gres=gpu:a100:1(S:0),gpu:1g.20gb:2(S:1),gpu:1g.10gb:3(S:1)
```

This tells you the node has one full A100 plus MIG slices of two different profiles.

### 5.5 GPU Batch Script Example

```bash
#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

module load miniconda3
source activate myenv

python train.py
```

> If no GPUs are requested (`--gres=gpu:0` or simply omitting `--gres`), no GPU time will be reserved or accounted for the job.

---

## 6. Environment Modules

Scientific software on Demetra is installed via the **Spack** package manager and made available through **environment modules**.

```bash
# List all available modules
module avail

# Show currently loaded modules
module show

# Load a module
module load <software-name>

# Unload all modules
module purge
```

---

## 7. Development Environments

### 7.1 RStudio Server (via Apptainer/Singularity)

A pre-built RStudio Server image is available at `/opt/rstudio`.

```bash
# 1. Get onto a compute node
srun -n1 --pty bash

# 2. Load apptainer
module load apptainer

# 3. Create working directories (first time only)
mkdir -p run state

# 4. Generate the launch command
singularity_rstudio_launcher --pass 'your_password'
```

This prints a full `singularity exec` command and the URL to connect to (e.g. `http://172.30.121.94:46155`). Open that URL in your browser (from the UniTS network/VPN), then log in with your username and the password you chose.

To also access your `/share` folder from RStudio, add an extra bind mount before the image name:

```bash
--bind /share/<group>/<username>:/share
```

### 7.2 Jupyter Lab (via Miniconda)

```bash
# 1. Get onto a compute node
srun -n1 --pty bash

# 2. Load miniconda and activate the jlab environment
module load miniconda3
source activate jlab

# 3. Launch Jupyter Lab
jupyter-lab --ip "$(hostname -I | awk '{print $1}')" --no-browser
```

Copy the printed URL (the one that is **not** `127.0.0.1`) into your browser. The URL includes an authentication token.

To access your `/share` folder, create a symlink in your home:

```bash
ln -s /share/<group>/<username> ~/shared
```

### 7.3 SageMath

```bash
module load sagemath

# CLI session
sage

# SageMath inside JupyterLab (with Sage kernels)
sage --notebook jupyterlab
```

For the JupyterLab mode, copy the printed URL and replace `localhost` with the node's IP (get it via `hostname -I`).

### 7.4 MATLAB

MATLAB requires a **personal license** (no cluster-wide license is available). After activating your license on MathWorks (using your Demetra Unix username and a MAC address from the target compute node), place the `license.lic` file at:

```
~/.matlab/R2022b_licenses/license.lic
```

Or export the path:

```bash
export MLM_LICENSE_FILE=/path/to/license.lic
```

Usage:

```bash
module load matlab/R2022b-singularity
matlab -batch "my_script"
```

MAC addresses for compute nodes (first interface):

| Node | MAC Address |
|---|---|
| babbage | `b0:26:28:21:c9:4e` |
| lovelace01 | `70:b5:e8:f0:14:7c` |
| lovelace02 | `70:b5:e8:f0:15:ec` |
| turing01 | `98:f2:b3:0e:0b:e0` |
| turing02 | `98:f2:b3:0e:05:28` |
| turing03 | `98:f2:b3:0e:13:bc` |

### 7.5 Apptainer / Singularity Builds

When building images, Apptainer uses `/tmp` (~56 GB). For larger builds, redirect the temp directory:

```bash
export APPTAINER_TMPDIR=/path/to/large/tmp
```

Note that writing to your home will be slow because it is NFS-mounted from the NetApp.

---

## 8. Practical Examples

### 8.1 Simple Serial Job

```bash
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --partition=Main
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=hello_%j.out

echo "Hello from $(hostname) at $(date)"
```

### 8.2 Multi-Core OpenMP Job

```bash
#!/bin/bash
#SBATCH --job-name=openmp_test
#SBATCH --partition=turing-wide
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=omp_%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./my_openmp_program
```

### 8.3 MPI Job

```bash
#!/bin/bash
#SBATCH --job-name=mpi_test
#SBATCH --partition=turing-wide
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=18
#SBATCH --time=02:00:00
#SBATCH --output=mpi_%j.out

module load openmpi
srun ./my_mpi_program
```

### 8.4 GPU Job with Python

```bash
#!/bin/bash
#SBATCH --job-name=gpu_python
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=gpu_%j.out
#SBATCH --error=gpu_%j.err

module load miniconda3
source activate myenv

python train.py --epochs 100 --batch-size 64
```

### 8.5 Long-Running Thin Job

```bash
#!/bin/bash
#SBATCH --job-name=long_sim
#SBATCH --partition=turing-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00          # 14 days (partition max)
#SBATCH --output=sim_%j.out

./my_long_simulation
```

### 8.6 Multiple Partitions

You can specify more than one partition to increase scheduling chances:

```bash
#SBATCH --partition=Main,lovelace
```

---

## 9. Quick Reference Card

```text
# Connect
ssh <ID>@demetra.units.it

# Check cluster status
sinfo

# Submit a job
sbatch script.sh

# Interactive session
srun -n1 --pty bash

# Interactive session with GPU
srun -n1 --gres=gpu:1 --pty bash

# Check your jobs
squeue -u $USER

# Cancel a job
scancel <job-id>

# Job history
sacct -j <job-id>

# List available software
module avail

# Load / unload software
module load <name>
module purge

# Check GPU info on a node
scontrol show node <nodename> | grep -i gres

# Your home directory
/u/<username>

# Your shared data
/share/<group>/<username>
```

---

## 10. Useful Links

- **Demetra Documentation:** [https://demetra.units.it/docs/](https://demetra.units.it/docs/)
- **SLURM Quick Start:** [https://slurm.schedmd.com/quickstart.html](https://slurm.schedmd.com/quickstart.html)
- **SLURM sbatch Reference:** [https://slurm.schedmd.com/sbatch.html](https://slurm.schedmd.com/sbatch.html)
- **Apptainer Documentation:** [https://apptainer.org/](https://apptainer.org/)
- **Environment Modules:** [http://modules.sourceforge.net/](http://modules.sourceforge.net/)
- **NVIDIA MIG Guide:** [https://developer.nvidia.com/blog/dividing-nvidia-a30-gpus-and-conquering-multiple-workloads](https://developer.nvidia.com/blog/dividing-nvidia-a30-gpus-and-conquering-multiple-workloads)
- **Support Email:** `support+migemigra@exact-lab.it` (Cc `mige.hpc@units.it`)
