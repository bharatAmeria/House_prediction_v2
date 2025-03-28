pipeline {
    agent any  // Run on any available agent

    environment {
        VENV_DIR = "tracker"  // Define virtual environment path
    }

    triggers {
        pollSCM('* * * * *')  // Polls the repo every minute for changes
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/bharatAmeria/House_prediction_v2.git'
            }
        }

        stage('Setup Environment') {
            steps {
                sh 'python3 -m venv $VENV_DIR'
                sh './tracker/bin/pip install -r requirements.txt'
            }
        }

        stage('Test Environment') {
            steps {
                sh './tracker/bin/python testEnvironment.py'
            }
        }

        stage('Recommender System') {
            steps {
                sh './tracker/bin/python runPipeline.py'
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed. Check logs.'
        }
    }
}
