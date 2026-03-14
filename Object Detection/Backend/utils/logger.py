"""
Professional Logging Configuration

Provides structured logging with different levels for different components.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured, readable logs"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Determine color
        color = self.COLORS.get(record.levelname, self.COLORS['INFO'])
        reset = self.COLORS['RESET']
        
        # Format message
        if record.levelname in ['ERROR', 'CRITICAL']:
            # Include exception info for errors
            msg = f"{color}[{timestamp}] [{record.levelname}] {record.name}: {record.getMessage()}{reset}"
            if record.exc_info:
                msg += f"\n{self.formatException(record.exc_info)}"
        else:
            msg = f"{color}[{timestamp}] [{record.levelname}] {record.name}: {record.getMessage()}{reset}"
        
        return msg


def setup_logging(
    name: str,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_console: bool = True
) -> logging.Logger:
    """
    Setup structured logging for a module
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory for log files
        console_level: Console output level
        file_level: File output level
        enable_console: Whether to enable console output
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Console Handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = StructuredFormatter()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File Handler (rotating logs)
    log_file = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
    
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(file_level)
        
        # File format is more detailed (no colors)
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except IOError as e:
        logger.warning(f"Could not setup file logging: {e}")
    
    return logger


# Global logger instance
_main_logger: Optional[logging.Logger] = None


def get_logger(name: str = "vision_system") -> logging.Logger:
    """Get or create the main logger"""
    global _main_logger
    
    if _main_logger is None:
        _main_logger = setup_logging(name)
    
    return _main_logger
