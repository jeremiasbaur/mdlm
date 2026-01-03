from setuptools import setup, find_packages

setup(
    name='mdlm',  # This is the name you'll use, e.g., "pip install mdlm"
    version='0.1.0',
    
    # This automatically finds all folders with an __init__.py 
    # (like your 'models', 'scripts', and 'configs' folders)
    # and includes them as installable sub-packages.
    packages=find_packages(),
    
    # This tells setup to also include the loose .py files
    # in your root directory, so you can import them directly.
    py_modules=[
        'diffusion',
    ]
)