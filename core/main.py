import sys
import os
import subprocess
import argparse
import time

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)
        self.saved_path = os.getcwd()

    def __enter__(self):
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)

# TODO: add more models here
MODELS = {
    'cyclegan': {
        'dir': 'models/cyclegan',
        'exe': 'cyclegan.py',
        'custArgs': [],
    },
    's2stransformer': {
        'dir': 'models/s2stransformer',
        'exe': 's2stransformer.py',
        'custArgs': ['--gpuIdx', '0', '--alpha', '0.0001', '--model_path', 's2s.pt'],
    }
}

def runModel(model, numSteps, batchSize, mode, logFile, asy=0):
    # construct command
    # using absolute for log file since we 'cd' into model dir
    absLogFile = os.path.abspath(logFile)
    cmd = [
        'python3', model['exe'],
        '--mode', mode,
        '--batch_size', str(batchSize),
        '--num_steps', str(numSteps),
        '--log_file', absLogFile,
        '--enable_perf_log'
    ]
    if model['custArgs']:
        cmd.extend(model['custArgs'])
    
    with cd(model['dir']):
        print(f"Starting process: {' '.join(cmd)}")
        if asy:
            # run asynchronously
            return subprocess.Popen(cmd)
        else:
            # run and wait
            return subprocess.run(cmd)

def main(args):
    # parse comma separated arguments
    modelNames = args.models.split(',')
    jobTypes = args.type.split(',')
    stepsList = list(map(int, args.num_steps.split(',')))
    batchSizes = list(map(int, args.batch_sizes.split(',')))
    logPath1 = args.log_path1
    logPath2 = args.log_path2

    selectedModels = []
    for name in modelNames:
        if name in MODELS:
            selectedModels.append(MODELS[name])
        else:
            print(f"Model {name} not recognized. Available models: {list(MODELS.keys())}")
            sys.exit(getattr(os, 'EX_USAGE', 64))
    
    # determine if we are colocating (async) or running serially
    asy = args.colocate == 1
    processes = []

    # start processes
    for i,m in enumerate(selectedModels):
        p = runModel(
            m,
            numSteps=stepsList[i],
            batchSize=batchSizes[i],
            mode=jobTypes[i],
            logFile=(logPath1 if i == 0 else logPath2),
            asy=asy
        )
        processes.append(p)
    
    # monitor processes if colocated
    if asy:
        print("Monitoring colocated processes...")
        finished = False
        while not finished:
            for p in processes:
                if p.poll() is not None: # one process finished
                    print(f"\nTerminating remaining processes after {p.args[1]} finished.")
                    for p2 in processes:
                        if p2.poll() is None:
                            p2.terminate()
                    finished = True
                    break
            time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Job Orchestrator for Co-location Analysis')

    parser.add_argument('--MPS', type=int, default=1, help='MPS status (managed by shell script)')
    parser.add_argument('-c', '--colocate', type=int, default=1, help='1 to enable co-location, 0 for serial')
    parser.add_argument('-m', '--models', type=str, required=True, help='Comma-separated list of models to run')
    parser.add_argument('-b', '--batch_sizes', type=str, required=True, help='Comma-separated list of batch sizes for each model')
    parser.add_argument('-n', '--num_steps', type=str, required=True, help='Comma-separated list of number of steps for each model')
    parser.add_argument('-t', '--type', type=str, default='training,inference', help='Comma-separated list of job types: training, inference')
    parser.add_argument('--log_path1', type=str, required=True, help='Path to log file 1')
    parser.add_argument('--log_path2', type=str, required=True, help='Path to log file 2')
    args = parser.parse_args()
    main(args)
                        