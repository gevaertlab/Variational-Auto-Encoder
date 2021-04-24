''' Evaluate the encoding with downstream tasks (applications) '''
from applications.application import Application

if __name__ == '__main__':
    log_name = 'VAE3D32'
    version = 48
    app = Application(log_name, version, task_name='task_volume')
    app.taskPrediction(models=['linear_regression'])
    app.saveResults()