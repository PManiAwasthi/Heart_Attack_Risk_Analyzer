from setuptools import setup, find_packages
from typing import List

PROJECT_NAME = "Heart-Risk-Analyzer"
AUTHOR = "PRAKASHMANI AWASTHI"
VERSION = "0.0.1"
DESCRIPTION = "This is full stack data science project for predict the risk/possibility of heart attack in the 10 years."

REQUIREMENT_FILE_NAME = 'requirements.txt'
HYPHEN_E_DOT = '-e .'

def get_requirements_list() -> List[str]:
    """
    Description: this function is to return a list of requirements
    mentioned in the requirements.txt file
    return of this function is a list containing all the libraries
    mentioned in the requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirements_file:
        requirements_list = requirements_file.readlines()
        requirements_list = [requirements_name.replace("\n", "") for requirements_name in requirements_list]
        requirements_list.remove(HYPHEN_E_DOT)
        return requirements_list

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements_list()
)