#
# TITLE: Python data visualization
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: python data visualization with matplotlib and seaborn, 2025 by Mike J. Maxwell
#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
# === chapter 1. foundation of data visualization
#

# === 1.2 the visualization process: from raw data to insight
if False:
    # load and inspect data
    tips = sns.load_dataset('tips')
    print(tips.head())
    print(tips.describe())
    # quick check for missin values
    print('missing values:\n', tips.isna().sum())
    # visualize relationship between total_bill and tip
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time', style='sex', s=70, ax=ax)
    ax.set_title('tip vs total bill by time and sex')
    ax.set_xlabel('total bill ($)')
    ax.set_ylabel('Tip ($)')
    plt.tight_layout()
    plt.show()

# === 1.3 choosing the right chart for the right data
if False:
    # good visualzation design begins with matching charts types to data types and analytical goals
    # when comparing two numeric variables, scatter plots work best
    # for distribution, historgram, boxplots, or violin plots highlight spread and central tendency
    # to analyze categorical differences, bar plots and boxplots are effective
    # when studying time trends, line charts are ideal
    # to understand correlations, heatmaps and pair plots reveal relationships among variables
    # examples:
    tips = sns.load_dataset('tips')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # histogram + KDE
    sns.histplot(data=tips, x='total_bill', bins=20, kde=True, ax=ax[0])
    ax[0].set_title('distribution of total bill')
    ax[0].grid(ls=':')
    # boxplot comparing lunch vs dinner
    sns.boxplot(data=tips, x='time', y='total_bill', ax=ax[1])
    ax[1].set_title('boxplot of total bill by time')
    ax[1].grid(ls=':')
    plt.tight_layout()
    plt.show()
    # different visual forms highlight different aspects of the same dataset
    # the histogram shows the distribution's shape,
    # while the boxplot compares two conditions

# === 1.4 color, clarity, and storytelling principles
if True:
    # effective visualization design relies on three key ideas
    # clarity, consistency, and purpose
    # clarity
    # simplicity beat complexity
    # reduce visual nose - avoid unnecessary lines, excessive colors or redundant labels
    # color
    # use color to encode meaning, not decoration
    # seaborn provides colorblind-friendly palettes like 'colorbind' and 'Set2'
    # storytelling
    # your visual should answer a question or make a point
    # titles and captions should summarize insights, not merely describe content
    #
    # consider the example blow
    # adding annotation transforms a static plot into a visual story
    tips = sns.load_dataset(name='tips')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=tips, x='total_bill', y='tip', hue='day', ax=ax)
    ax.set_title('tip vs total bill with outlier annotation')
    ax.grid(ls=':')
    plt.show()
    
