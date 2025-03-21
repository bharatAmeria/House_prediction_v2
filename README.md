# House Price Prediction 

# House Price Prediction App

## Project Summary
The House Price Prediction App is designed to scrape real estate data from [99acres.com](https://www.99acres.com) and utilize machine learning techniques to predict house prices accurately for the city gurgaon. The captured data is stored in Google Drive, ensuring accessibility and scalability. The project follows a structured ML pipeline where data ingestion are efficiently managed. The entire process, including the storage of artifacts, is systematically maintained within the project.

## Initial Setup
To set up and run the project, follow these steps:

### Prerequisites
- Python (>= 3.8)
- pip package manager
- Required Python libraries

### Steps to Setup

1. **Create a Virtual Environment**
   ```sh
   python -m venv venv
   source venv/bin/activate 
   
   On Windows use: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Test Environment**
   ```sh
   python testEnvironment.py
   ```

4. **Run Pipeline**
   ```sh
   python runPipeline.py
   ```