from setuptools import setup, find_packages

with open('README.md', 'r') as f:
  long_description = f.read()

setup(
  name='gym-games',
  version='1.0.4',
  keywords=['AI', 'Reinforcement Learning', 'Games', 'Pygame', 'MinAtar'],
  description='This is a gym version of various games for reinforcement learning.',
  url='https://github.com/qlan3/gym-games',
  author='qlan3',
  author_email='qlan3@ualberta.ca',
  license='MIT',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=find_packages(),
  python_requires='>=3.5',
  install_requires=[
    'numpy',
    'MinAtar',
    'gymnasium',
    'setuptools',
    'pygame',
    'ple'
  ]
)
