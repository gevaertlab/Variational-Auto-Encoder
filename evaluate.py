''' Evaluate the encoding with downstream tasks (applications) '''
import argparse
from __init__ import TASK_NAMES
from applications.application import Application

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description='VAE Downstream Tasks Evaluation')
    parser.add_argument('--log-name',
                        default='VAE3D32AUG', type=str,
                        help="Name of the trained model/directory of saved log")
    parser.add_argument('--version',
                        default=10, type=int,
                        help="Version number of the saved log")
    parser.add_argument('--tasks', nargs='+', type=str,
                        default='all',
                        help="name of tasks to run")
    parser.add_argument('--models', nargs='+', type=str,
                        default='all',
                        help="name of models to run")
    parser.add_argument('--command', nargs='+', type=str,
                        default='both',
                        help='<task_predict> or <visualize> or <both>')
    args = parser.parse_args()

    if args.tasks == 'all':
        task_names = TASK_NAMES
    elif not all(t in TASK_NAMES for t in args.tasks):
        raise ValueError(
            f"{str(set(TASK_NAMES) - set(args.tasks))} not in known task names"
            )
    else:
        task_names = [args.tasks]

    for task_name in task_names[1:]:
        app = Application(args.log_name, args.version, task_name=task_name)
        if 'both' in args.command or 'task_predict' in args.command:
            app.task_prediction(tune_hparams=False, models=args.models)
            # app.save_results()
            app.draw_dignosis_figure()
        if "both" in args.command or "visualize" in args.command:
            app.visualize()
    pass
