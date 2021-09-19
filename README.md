[![build-n-test](https://github.com/lucasSaavedra123/StatisticFrame/actions/workflows/main.yml/badge.svg)](https://github.com/lucasSaavedra123/StatisticFrame/actions/workflows/main.yml)

# statisticFrame
<p align="center">
  <img align="center" src="assets/logo.png">
</p>
StatisticFrame is a set of classes and tools that use most used machine learning and statistics libraries. Why? We take advantage of OOP concepts that those libraries don't. Use it. It's very comfortable and declarative.

# Requirements

What do we need? Just Python 3.
First, we need to install all required libraries. To do that, type:

    pip install -r requirements.txt

Finally, write:

    pip install git+https://github.com/lucasSaavedra123/StatisticFrame.git

Dockerization is a great idea. We could develop an API to be worldwide used for cloud services and stuff like that. But by now, it's used in this way.

# FAQ

## How do I run tests?

Move to `tests` folder and type

    python -m unittest discover -s . -p 'test_*.py' -v -b

## How do I uninstall it?

Easy, just type:

    pip uninstall statisticframe


# Contribution

Do you want to contribute? Follow the next steps:

1. **Create an issue**: If you have an idea to add in statisticframe, write an issue in GitHub and describe it (this is optional). In case you work on your own issue, assign it to you. If you can't do it and you want to give the idea, don't worry, just create the issue and someone will be working on it.
2. **Create a branch**: If you are going to work with an issue, create a branch with the following format: `yourUserName-issue-issueNumber`. 
3. **Work on that branch**.
4. **Make a Pull request**: Once your job is finished, make a pull request and statisticframe staff will review it. If everthing is fine, it will be merged into master. Otherwise, changes will be requested.

Please, work with any methodology that uses unit tests. The pull will not be approved unless tests are added. However, if testing is not required, it could be merged into master, but only if it is properly justified.

# Thanks to
<p align="center">
  <img align="center" src="assets/logo_uca.png">
</p>
