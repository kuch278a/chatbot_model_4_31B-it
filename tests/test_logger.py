import os
import unittest
from src.utils.logger import get_logger, LOGS_DIR

class TestLogger(unittest.TestCase):
    """Unit test for logs directory creation and logging functionality."""

    def test_logs_directory_exists(self):
        """Verify logs/ directory exists."""
        self.assertTrue(os.path.exists(LOGS_DIR))
        self.assertTrue(os.path.isdir(LOGS_DIR))

    def test_logger_writes_to_log_file(self):
        """Verify logger writes formatted log messages to logs/app.log."""
        logger = get_logger("test_module")
        test_msg = "Test log entry for Amani AI assistant"
        logger.info(test_msg)

        log_file_path = os.path.join(LOGS_DIR, "app.log")
        self.assertTrue(os.path.exists(log_file_path))

        with open(log_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn(test_msg, content)

if __name__ == "__main__":
    unittest.main()
