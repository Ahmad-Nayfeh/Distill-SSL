import pandas as pd
import os

class Logger:
    """
    A simple logger class that saves results to a CSV file.
    """
    def __init__(self, filepath, header):
        """
        Args:
            filepath (str): Path to the CSV file (e.g., 'logs/B1_student.csv')
            header (list): List of column names (e.g., ['epoch', 'train_loss', 'test_acc'])
        """
        self.filepath = filepath
        self.header = header
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create the file and write the header
        self.df = pd.DataFrame(columns=header)
        self.df.to_csv(self.filepath, index=False)

    def log(self, data_dict):
        """
        Logs a new row of data.
        
        Args:
            data_dict (dict): A dictionary of data (e.g., {'epoch': 1, 'train_loss': 0.5})
        """
        # Append data to the CSV file
        # This is not the most efficient for huge files, but perfect for our scale.
        with open(self.filepath, 'a') as f:
            self.df = pd.DataFrame(data_dict, index=[0])
            self.df.to_csv(f, header=False, index=False)

if __name__ == '__main__':
    # Test the logger
    test_log_path = 'logs/test_log.csv'
    header = ['epoch', 'loss', 'acc']
    logger = Logger(filepath=test_log_path, header=header)
    
    logger.log({'epoch': 1, 'loss': 0.5, 'acc': 0.8})
    logger.log({'epoch': 2, 'loss': 0.4, 'acc': 0.85})
    
    print(f"Test log created at {test_log_path}")
    df = pd.read_csv(test_log_path)
    print(df)