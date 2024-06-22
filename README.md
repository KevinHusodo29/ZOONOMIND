# ZOONOMIND  

## Introduction  

The system developed in this research, called ZOONOMIND, is an early detection platform for zoonotic diseases based on machine learning technology designed to enhance global health responses to potential future outbreaks. By collecting and integrating data from various sources such as the environment, animals, human health, and epidemiology in real-time, ZOONOMIND enables rapid analysis of disease patterns and prediction of potential outbreaks. This system analyzes disease spread patterns, identifies habits, seasonal patterns, and risk factors using machine learning algorithms. The analysis results are presented visually through a geographic information system (GIS), mapping disease spread patterns spatially. The user-friendly interface allows effective access to information. Currently, ZOONOMIND can detect and analyze the spread of Dengue Fever, enabling real-time monitoring of spread patterns, identification of high-risk areas, prediction of potential outbreaks, and presentation of interactive maps for quick response by authorities.  

## System Design  

The software will be built using the Python programming language. Data collection, analysis, and processing will also utilize Python. The following are the main features of ZOONOMIND:  

- Early Detection: ZOONOMIND will be able to monitor community health conditions that can serve as early indicators of potential Dengue Fever outbreaks.  
- Data Analysis: ZOONOMIND will employ big data analytics technology to analyze data from various sources and identify patterns and risk factors that could potentially trigger disease spread.

## Instructions  

- Environment and File Preparation:
    
  + Ensure your Python environment is set up with necessary modules like Streamlit, pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, plotly, and opencage.
  + Download the Dengue-worldwide-dataset-modified.xlsx dataset and place it in the appropriate directory.

- Running the Streamlit Application:

  + Navigate to the directory where your Python script (nama_file.py) containing the Streamlit application code is located.  
  + If you're using a virtual environment, activate it first to ensure dependencies are correctly loaded. This step depends on your virtual environment setup (e.g., source venv/bin/activate for Unix/Linux, venv\Scripts\activate for Windows).  
  + Ensure all required Python packages (e.g., Streamlit, pandas, scikit-learn) are installed in your environment. If not, install them using pip.  
  + Once in the correct directory and with your virtual environment activated (if applicable), use the following command to run your Streamlit application.  
  + After executing the command, Streamlit will start a local web server and provide a URL (usually http://localhost:8501) where you can view your application.
