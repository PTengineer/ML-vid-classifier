import unittest
import logging
import sys
import os
import shutil
from io import StringIO

# Mock the cfg object and its log_dir attribute minimally
class MockConfig:
    """Minimal mock for the configuration object."""
    # Define a temporary directory for logging tests
    LOG_DIR = './temp_log_tests/'
    log_dir = LOG_DIR

# IMPORTANT: Mock the config used in logger.py
# In a real setup, you would use unittest.mock.patch to replace the imported cfg.
# For simplicity here, we assume logger.py is accessible for testing.
from logger import get_logger 
import logger # Import the module to patch its cfg

# Temporarily set the logger's cfg to our mock for testing isolation
logger.cfg = MockConfig()


class TestLoggerSetup(unittest.TestCase):
    
    LOG_FILENAME = "test_log_output.txt"
    LOG_PATH = os.path.join(MockConfig.LOG_DIR, LOG_FILENAME)

    def setUp(self):
        """Prepares the testing environment before each test."""
        # Ensure the mock log directory exists
        os.makedirs(MockConfig.LOG_DIR, exist_ok=True)
        # Clear the logger handlers to prevent cross-contamination between tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        
    def tearDown(self):
        """Cleans up the testing environment after each test."""
        # Remove the temporary log directory and all files within
        if os.path.exists(MockConfig.LOG_DIR):
            shutil.rmtree(MockConfig.LOG_DIR)


    def test_01_file_handler_creation_and_content(self):
        """Test if the log message is correctly written to the file."""
        test_message = "Test message for file handler integrity."
        
        # 1. Get and use the logger
        log = get_logger("FileTest", self.LOG_FILENAME)
        log.info(test_message)
        
        # 2. Force flush and close handlers to ensure data is written to disk
        for handler in log.handlers:
            handler.flush()
        
        # 3. Verify the file exists and content is correct
        self.assertTrue(os.path.exists(self.LOG_PATH))
        
        with open(self.LOG_PATH, 'r') as f:
            content = f.read()
            self.assertIn(test_message, content)
            self.assertIn("FileTest", content) # Check logger name is present
            
            
    def test_02_console_handler_warning_output(self):
        """Test if WARNING messages go to console (stdout)."""
        test_warning = "Critical training complete."
        test_info = "Epoch 1 detailed metrics."
        
        # Redirect stdout to capture printed output
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # 1. Get and use the logger
            log = get_logger("ConsoleTest", self.LOG_FILENAME)
            
            # This should print to console (WARNING level)
            log.warning(test_warning) 
            
            # This should NOT print to console (INFO level)
            log.info(test_info)
            
            # 2. Capture the output
            captured_output = sys.stdout.getvalue()
        finally:
            # Restore stdout regardless of test outcome
            sys.stdout = original_stdout

        # 3. Verify console output
        # Check if the WARNING message is present
        self.assertIn(test_warning, captured_output) 
        
        # Check if the INFO message is absent (proving the filter works)
        self.assertNotIn(test_info, captured_output)
        
        
    def test_03_no_duplicate_handlers(self):
        """Test that calling get_logger multiple times doesn't add duplicate handlers."""
        log_name = "DuplicateTest"
        
        # Call 1: Sets up the handlers
        log1 = get_logger(log_name, self.LOG_FILENAME)
        num_handlers_initial = len(log1.handlers)
        
        # Call 2: Should return the existing logger without adding new handlers
        log2 = get_logger(log_name, self.LOG_FILENAME)
        num_handlers_final = len(log2.handlers)
        
        # We expect 2 handlers (FileHandler and StreamHandler)
        self.assertEqual(num_handlers_initial, 2)
        # We expect the count to be the same after the second call
        self.assertEqual(num_handlers_initial, num_handlers_final)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)