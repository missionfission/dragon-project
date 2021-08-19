## DRAGON Framework Features :

1. Runs on any Pytorch Model and provides the Graphical Analysis of Execution
2. Fast Hardware Performance Estimation (Runtime, Energy and Area), allows overlapped execution and prefetching
3. Rapid Automatic Hardware Design over a Set of Dataflow Graph, and the corresponding Scheduling Algorithm
4. Allow Specifying Different Optimization and Constraints
5. Evaluate Emerging Technologies (RRAM, CNFETs), Next Computing and Architectures (In-Memory, 3D SoCs)


---

## Installation

To get started from github, install the Google Colab Plugin : https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo?hl=en

And, click the Colab Chrom Plugin, when you open the Demo.ipynb on github.

The Demo provides running of all the different features of our Framework : 1) Performance Estimation 2) Hardware Architecture Optimization and Synthesis and 3) Technologies Target Derivations 

---

For manual installation on PC, install Miniconda3 : https://docs.conda.io/en/latest/miniconda.html (Python 3.6)

Create a Conda environment

---
conda create -n perf

conda activate perf 

conda install pytorch torchvision torchaudio cpuonly -c pytorch

---

Open Demo.ipynb in jupyter notebook

---
Performance Estimation for AI and Non-AI Workloads :

1. Runs on Pytorch Model of an AI Workload and provides the Graphical Analysis of Execution
---

For Running the Pytorch see models specified as in 'src/common_models.py', the default-hardware configurations supported are specified in 'src/configs/default.yaml'
To specify the configs completely, detail all the energy and timing details for input to the simulation.

---


2. For Running Non-AI Workload in C++, we have to get the LLVM-Trace :

We have provided Generate LLVM-Trace for some applications in folder 'req/.."
The application trace is given in form of .gz and is generated from LLVM 6.0,

We use an open-source LLVM-Trace Generation for doing this : https://github.com/harvard-acc/LLVM-Tracer/tree/llvm-6.0, place the workload to run in the folder "src/req/your_app", with a cfg and trace file in "src/req/your_app/inputs/" directory

Add the name of your app to src/ddfg_main.py which -> calls the C++ Non-AI synthesis Code. 

---


3. For Running Non-AI Workload in Python, we have to got the Python HLS as follows :

In the synthesis folder go to "src/hls.py", and call the gen_stats and cfg generation functions

