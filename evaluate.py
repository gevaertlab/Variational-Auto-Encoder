''' Evaluate the encoding with downstream tasks (applications) '''
from applications.application import Application

if __name__ == '__main__':
    log_name = 'VAE3D32'
    version = 48
    task_names = [
        #   'task_volume',
        # 'task_malignancy',
        # 'task_texture',
        # 'task_spiculation',
        'task_subtlety'
    ]
    
    for task_name in task_names:
        app = Application(log_name, version, task_name=task_name)
        app.taskPrediction(models='all')
        app.saveResults()
        app.draw_dignosis_figure()

    # app = Application(log_name, version, task_name='task_malignancy')
    # app.taskPrediction(models=['logistic_regression'])
    # app.saveResults()
    # app.draw_dignosis_figure()
