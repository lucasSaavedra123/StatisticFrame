import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='statisticframe',
    package_dir={"statisticframe": "src"},
    author='Lucas Ariel Saavedra',
    author_email='lucasarielsaavedra@hotmail.com',
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lucasSaavedra123/StatisticFrame',
    project_urls={"StatisticFrame repository": "https://github.com/lucasSaavedra123/StatisticFrame"},
    packages=["statisticframe.Utils","statisticframe.Algorithm", "statisticframe.Model"]
)
