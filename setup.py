from setuptools import setup, find_packages

setup(
    name="my_package",  # The name of your package
    version="0.0.1",  # Your package version
    packages=find_packages(),  # Automatically find package directories
    install_requires=[  # List your dependencies here (optional)
        # 'numpy',  # Example: Add any external libraries you need
    ],
    include_package_data=True,  # Include files listed in MANIFEST.in
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your package",
    long_description=open('README.md').read(),  # Use README.md as the long description
    long_description_content_type="text/markdown",  # For markdown README
    url="https://github.com/yourusername/my_package",  # Your project URL (e.g., GitHub)
    classifiers=[  # Optional: to categorize your package on PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',  # Specify the minimum Python version required
    entry_points={  # Optional: For command-line tools
        'console_scripts': [
            'my_package-cli=my_package.cli:main',  # Example: my_package-cli command
        ],
    },
)
