import sys
import subprocess

def run_task(task_number):
    task_file = f'task{task_number}.py'
    subprocess.run(['python', task_file])


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python A3.py <TASK#>")
        sys.exit(1)

    task_number = sys.argv[1]
    run_task(task_number)
