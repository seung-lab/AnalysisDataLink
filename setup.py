from setuptools import setup
import re
import os
import codecs
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
   name='AnalysisDataLink',
   version='0.1.0',
   description='Tools to query data from the dynamic annotation framework',
   long_description=open('README.md').read(),
   author='Forrest Collman, Sven Dorkenwald, Casey Schneider-Mizell',
   author_email='caseysm@gmail.com',
   url="https://github.com/seung-lab/AnalysisDataLink",
   packages=find_packages(), 
   install_requires=[required]
)
