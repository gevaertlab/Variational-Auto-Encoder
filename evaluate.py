''' Evaluate the encoding with downstream tasks (applications) '''
import argparse
from __init__ import TASK_NAMES
from applications.application import Application

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description='VAE Downstream Tasks Evaluation')
    parser.add_argument('--log-name',
                        default='VAE3D32', type=str,
                        help="Name of the trained model/directory of saved log")
    parser.add_argument('--version',
                        default=70, type=int,
                        help="Version number of the saved log")
    parser.add_argument('--tasks', nargs='+', type=str,
                        default='all',
                        help="name of tasks to run")
    parser.add_argument('--models', nargs='+', type=str,
                        default='all',
                        help="name of models to run")
    args = parser.parse_args()

    if args.tasks == 'all':
        task_names = TASK_NAMES
    elif not all(t in TASK_NAMES for t in args.tasks):
        raise ValueError(
            f"{str(set(TASK_NAMES) - set(args.tasks))} not in known task names")

    for task_name in task_names:
        app = Application(args.log_name, args.version, task_name=task_name)
        # TODO: implement flexible model selection
        app.taskPrediction(models=args.models)
        app.saveResults()
        app.draw_dignosis_figure()
        app.visualize()
