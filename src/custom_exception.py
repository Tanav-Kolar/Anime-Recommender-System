import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: sys) -> str:

        _, _, exc_tb = error_detail.exc_info()
        line_number = exc_tb.tb_lineno if exc_tb else 'Unknown'
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else 'Unknown'
        detailed_message = f"Error occurred in script: {file_name} at line: {line_number} with message: {error_message}"
        return detailed_message
    
    def __str__(self):
        return self.error_message