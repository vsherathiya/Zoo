
from setuptools import setup
import setuptools

with open('README.md', 'r', encoding='utf8-') as f:
    long_description = f.read()


__version = '0.0.0'

REPO_NAME = 'MitraAI-API'
AUTHOR_USER_NAME = 'AmitNexuslink'
SRC_REPO = 'MitraAI-API'
AUTHOR_EMAIL = '.'

setup(
    name=SRC_REPO,
    version=__version,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='Mitra-API',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src')
)