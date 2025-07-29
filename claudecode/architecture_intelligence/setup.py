#!/usr/bin/env python3
"""
Setup script for Architecture Intelligence Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]

setup(
    name="architecture-intelligence",
    version="1.0.0",
    description="Deep architecture framework expertise with intelligent pragmatism",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Architecture Intelligence Team",
    author_email="team@architecture-intelligence.com",
    url="https://github.com/your-org/architecture-intelligence",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "mypy>=1.0.0",
            "coverage>=6.0",
            "pre-commit>=2.20.0"
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0", 
            "mkdocs-mermaid2-plugin>=0.6.0"
        ],
        "enterprise": [
            "ldap3>=2.9.0",
            "pyodbc>=4.0.0",
            "kubernetes>=24.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "arch-intel=architecture_intelligence.cli:cli",
            "ai-arch=architecture_intelligence.cli:cli",
            "architecture-intelligence=architecture_intelligence.cli:cli"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "architecture", "enterprise-architecture", "togaf", "ddd", "c4-model",
        "zachman", "archimate", "microservices", "patterns", "intelligence",
        "ai", "framework", "analysis", "design", "modeling"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/architecture-intelligence/issues",
        "Source": "https://github.com/your-org/architecture-intelligence",
        "Documentation": "https://architecture-intelligence.readthedocs.io/",
    },
    package_data={
        "architecture_intelligence": [
            "templates/**/*",
            "schemas/**/*",
            "configs/**/*"
        ]
    }
)