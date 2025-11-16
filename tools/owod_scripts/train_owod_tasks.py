import os.path as osp
import glob
from pathlib import Path
import subprocess


owod_settings = {
    "MOWODB": {
        "task_list": [0, 20, 40, 60, 80],
        "test_image_set": "all_task_test"
    },
    "SOWODB": {
        "task_list": [0, 19, 40, 60, 80],
        "test_image_set": "test",
    },
    "nuOWODB": {
        "task_list": [0, 10, 17, 23],
        "test_image_set": "test",
    }
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train OWOD tasks')
    parser.add_argument('dataset', type=str, choices=["MOWODB", "SOWODB", "nuOWODB"])
    parser.add_argument('config', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--start', type=int, default=1, help='Start task number')
    parser.add_argument('--threshold', type=float, default=0.05, help='Confidence score threshold for known class')
    parser.add_argument('--suffix', type=str, help='Suffix for work_dir')
    parser.add_argument('--save', action='store_true', help='Save evaluation results to eval_output.txt')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs to use for training')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3', help='Comma-separated GPU indices to use, e.g., "0,1"')
    return parser.parse_args()


def run_command(command):
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end="")
    for line in process.stderr:
        print(line, end="")
    return_code = process.wait()
    return return_code


def run_dataset(dataset, config, load_from, args):
    stem = Path(config).stem
    task_num = len(owod_settings[dataset]['task_list'])
    # task_num = 2
    prev_work_dir = ''
    # args.start = 2
    for task in range(args.start, task_num):
        work_dir = f'work_dirs/{stem}_{dataset.lower()}_train_task{task}'
        if args.suffix:
            work_dir += f"_{args.suffix}"

        command = (f'CUDA_VISIBLE_DEVICES={args.gpu_ids} DATASET={dataset} TASK={task} THRESHOLD={args.threshold} SAVE={args.save} '
                   f'./tools/dist_train_owod.sh {config} {args.gpus} --amp --work-dir {work_dir}')

        if task > 1:
            if not prev_work_dir:
                prev_work_dir = f'work_dirs/{stem}_{dataset.lower()}_train_task{task-1}'
            # load_from = sorted(glob.glob(osp.join(prev_work_dir, "best*.pth")))[-1]
            load_from = sorted(glob.glob(osp.join(prev_work_dir, "epoch*.pth")))[-1]

        if load_from:
            command += f" --cfg-options load_from={load_from} "

        if args.save:
            with open('eval_outputs.txt', 'a') as f:
                f.write(f"{dataset} [Task {task}] (thresh: {args.threshold}) - {stem}\n")

        print("<TRAIN>:", command)
        return_code = run_command(command)
        if return_code != 0:
            print(f"Task {task} failed with return code {return_code}")
            break

        # Update prev_work_dir
        prev_work_dir = work_dir

        
if __name__ == '__main__':
    import sys

    # print(sys.executable)
    # print(sys.path)
    args = parse_args()
    # Run all tasks for the dataset
    run_dataset(args.dataset, args.config, args.ckpt, args)
