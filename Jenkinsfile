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

        stage('Data Ingestion') {
            steps {
                sh './tracker/bin/python src/pipeline/stage01_data_ingestion.py'
            }
        }

        stage('Data Pre Processing') {
            steps {
                sh './tracker/bin/python src/pipeline/stage02_data_cleaning.py'
            }
        }

        stage('Feature Selection') {
            steps {
                sh './tracker/bin/python src/pipeline/stage03_feature_selection.py'
            }
        }

        stage('Data Visualization') {
            steps {
                sh './tracker/bin/python src/pipeline/stage04_data_visualiztion.py'
            }
        }

        stage('Recommender System') {
            steps {
                sh './tracker/bin/python src/pipeline/stage05_recommender_system.py'
            }
        }

        stage('Model Training') {
            steps {
                sh './tracker/bin/python src/pipeline/stage06_model_training.py'
            }
        }

        // stage('Build Docker Image') {
        //     steps {
        //         script {
        //             sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
        //         }
        //     }
        // }

        // stage('Run Docker Container') {
        //     steps {
        //         script {
        //             // Stop & remove any existing container before starting a new one
        //             sh "docker stop ${CONTAINER_NAME} || true && docker rm ${CONTAINER_NAME} || true"

        //             // Run container
        //             sh "docker run -d --name ${CONTAINER_NAME} -p 8081:8081 ${IMAGE_NAME}:${IMAGE_TAG}"
        //         }
        //     }
        // }

        // stage('Verify Running Container') {
        //     steps {
        //         script {
        //             sh "docker ps -a"
        //         }
        //     }
        // }
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
