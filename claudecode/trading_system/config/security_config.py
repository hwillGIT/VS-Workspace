"""
Secure Configuration Manager

Handles secure loading of environment variables and sensitive configuration.
Replaces hardcoded secrets with secure environment-based configuration.
"""

import os
import hashlib
import secrets
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class SecurityConfigError(Exception):
    """Raised when security configuration is invalid or missing"""
    pass

class SecureConfigManager:
    """
    Secure configuration manager that loads sensitive data from environment variables
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize secure configuration manager
        
        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.env_file = env_file or self.project_root / ".env"
        
        # Load environment variables
        self._load_environment()
        
        # Validate required configuration
        self._validate_configuration()
        
        logger.info("Secure configuration loaded successfully")
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        if self.env_file.exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
        else:
            logger.warning(f".env file not found at {self.env_file}")
            # Create example .env file
            self._create_example_env_file()
    
    def _create_example_env_file(self):
        """Create example .env file with secure defaults"""
        example_content = """# Trading System Environment Variables
# SECURITY NOTICE: Replace all values with actual secure values

TRADING_API_KEY=your_secure_trading_api_key_here
DATA_FEED_API_KEY=your_secure_data_feed_api_key_here
DATABASE_PASSWORD=your_secure_database_password_here
ENCRYPTION_SECRET=your_32_character_encryption_key_here
JWT_SECRET_KEY=your_jwt_secret_key_minimum_32_characters_long
"""
        
        example_file = self.project_root / ".env.example"
        with open(example_file, 'w') as f:
            f.write(example_content)
        
        logger.warning(f"Created example .env file at {example_file}")
        logger.warning("Please copy .env.example to .env and update with actual values")
    
    def _validate_configuration(self):
        """Validate that required configuration is present and secure"""
        required_vars = [
            'TRADING_API_KEY',
            'DATABASE_PASSWORD',
            'ENCRYPTION_SECRET',
            'JWT_SECRET_KEY'
        ]
        
        missing_vars = []
        weak_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            elif len(value) < 16:  # Minimum security requirement
                weak_vars.append(var)
            elif value.startswith(('your_', 'replace_', 'change_')):
                weak_vars.append(var)
        
        if missing_vars:
            raise SecurityConfigError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
        
        if weak_vars:
            logger.warning(f"Weak configuration detected for: {', '.join(weak_vars)}")
            logger.warning("Please update these with secure values")
    
    def get_api_key(self, service: str) -> str:
        """
        Get API key for a specific service
        
        Args:
            service: Service name (e.g., 'trading', 'data_feed', 'bloomberg')
            
        Returns:
            str: API key
            
        Raises:
            SecurityConfigError: If API key is not configured
        """
        key_map = {
            'trading': 'TRADING_API_KEY',
            'data_feed': 'DATA_FEED_API_KEY',
            'broker': 'BROKER_API_SECRET',
            'bloomberg': 'BLOOMBERG_API_KEY',
            'reuters': 'REUTERS_API_KEY'
        }
        
        env_var = key_map.get(service)
        if not env_var:
            raise SecurityConfigError(f"Unknown service: {service}")
        
        api_key = os.getenv(env_var)
        if not api_key:
            raise SecurityConfigError(f"API key not configured for service: {service}")
        
        return api_key
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration
        
        Returns:
            Dict with database connection parameters
        """
        return {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', '5432')),
            'database': os.getenv('DATABASE_NAME', 'trading_system'),
            'user': os.getenv('DATABASE_USER', 'trading_user'),
            'password': os.getenv('DATABASE_PASSWORD'),
        }
    
    def get_encryption_key(self) -> bytes:
        """
        Get encryption key for sensitive data
        
        Returns:
            bytes: 32-byte encryption key
        """
        key_string = os.getenv('ENCRYPTION_SECRET')
        if not key_string:
            raise SecurityConfigError("Encryption secret not configured")
        
        # Derive a consistent 32-byte key from the secret
        return hashlib.sha256(key_string.encode()).digest()
    
    def get_jwt_secret(self) -> str:
        """
        Get JWT secret key
        
        Returns:
            str: JWT secret key
        """
        jwt_secret = os.getenv('JWT_SECRET_KEY')
        if not jwt_secret:
            raise SecurityConfigError("JWT secret key not configured")
        
        return jwt_secret
    
    def get_redis_config(self) -> Dict[str, Any]:
        """
        Get Redis configuration
        
        Returns:
            Dict with Redis connection parameters
        """
        return {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'password': os.getenv('REDIS_PASSWORD'),
            'decode_responses': True
        }
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token
        
        Args:
            length: Token length in bytes
            
        Returns:
            str: Hex-encoded secure token
        """
        return secrets.token_hex(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """
        Hash password using secure algorithm (PBKDF2 with SHA-256)
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
            
        Returns:
            tuple: (hashed_password, salt_hex)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 with SHA-256 (much more secure than MD5)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        
        return hashed.hex(), salt.hex()
    
    def verify_password(self, password: str, hashed_password: str, salt_hex: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password
            hashed_password: Hex-encoded hashed password
            salt_hex: Hex-encoded salt
            
        Returns:
            bool: True if password matches
        """
        salt = bytes.fromhex(salt_hex)
        expected_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        
        return secrets.compare_digest(expected_hash.hex(), hashed_password)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'
    
    def get_log_level(self) -> str:
        """Get logging level"""
        return os.getenv('LOG_LEVEL', 'INFO')
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            'max_workers': int(os.getenv('MAX_WORKERS', '4')),
            'cache_ttl': int(os.getenv('CACHE_TTL', '3600')),
            'batch_size': int(os.getenv('BATCH_SIZE', '1000'))
        }
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get risk management limits"""
        return {
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '100000')),
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '50000')),
            'var_confidence_level': float(os.getenv('VaR_CONFIDENCE_LEVEL', '0.95'))
        }

# Global instance for easy access
_config_manager = None

def get_config_manager() -> SecureConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SecureConfigManager()
    return _config_manager

def get_api_key(service: str) -> str:
    """Convenience function to get API key"""
    return get_config_manager().get_api_key(service)

def get_database_config() -> Dict[str, Any]:
    """Convenience function to get database config"""
    return get_config_manager().get_database_config()

def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data using secure algorithm
    
    Args:
        data: Sensitive data to hash
        
    Returns:
        str: Securely hashed data
    """
    # Use SHA-256 instead of MD5
    return hashlib.sha256(data.encode()).hexdigest()

def generate_secure_id() -> str:
    """
    Generate cryptographically secure ID
    
    Returns:
        str: Secure random ID
    """
    return secrets.token_urlsafe(32)

# Example usage and migration guide
if __name__ == "__main__":
    print("Secure Configuration Manager - Usage Example")
    print("=" * 50)
    
    try:
        config = SecureConfigManager()
        
        print("✓ Configuration loaded successfully")
        print(f"✓ Environment: {'Production' if config.is_production() else 'Development'}")
        print(f"✓ Log Level: {config.get_log_level()}")
        
        # Example API key usage (safe to display length only)
        try:
            trading_key = config.get_api_key('trading')
            print(f"✓ Trading API Key: {'*' * (len(trading_key) - 8) + trading_key[-8:]}")
        except SecurityConfigError as e:
            print(f"⚠ Trading API Key: {e}")
        
        # Example database config
        db_config = config.get_database_config()
        print(f"✓ Database: {db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        
        # Example secure operations
        token = config.generate_secure_token()
        print(f"✓ Generated secure token: {token[:16]}...")
        
        # Example password hashing
        test_password = "test_password_123"
        hashed, salt = config.hash_password(test_password)
        print(f"✓ Password hashing: {hashed[:32]}...")
        
        verified = config.verify_password(test_password, hashed, salt)
        print(f"✓ Password verification: {verified}")
        
    except SecurityConfigError as e:
        print(f"✗ Configuration Error: {e}")
        print("\nTo fix this:")
        print("1. Copy .env.example to .env")
        print("2. Update .env with your actual secure values")
        print("3. Ensure all API keys are at least 16 characters long")