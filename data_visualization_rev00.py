#
# TITLE: Python data visualization
# AUTHOR: Hyunseung Yoo
# PURPOSE:
# REVISION:
# REFERENCE: python data visualization with matplotlib and seaborn, 2025 by Mike J. Maxwell
#

import numpy as np
import scipy as sc
import sympy as sy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
# === chapter 1. foundation of data visualization
#
# visualization is where analysis meets communication
# good visuals you uncover structure in data and communicate insights persuasively
# in this chapter, we'll explore why visualization matters in analytics and machine learning,
# how to transform raw data into insight, how to choose the right chart for your data,
# and the essential design principles behind effective storytelling
# we'll close with an introdunction to Matplotlib and Seaborn
# the two visualization libraries that power this book

# === 1.1 understanding the role of visualization in analytics and ML
#
# in analytics and machine learning, visualization is more than aesthetics - it's a method of thinking
# models are only as good as the understanding behind them,
# and visualization is often the quickest way to build that understanding
# 
# visualization supports three major activities:
# 
# exploration
# quick plots reveal data distributions, detect outliers, and expose correlations
# diagonals
# visualization helps identify model issues overfitting, underfitting, or data drift
# long before they become performance problems
# communication
# visual storytelling allows you to share complex results with non-technical audiences
# in ways they can grasp instantly
#
# a simple  scatter plot can reveal more about your data's structure than pages of statistics
# in one object, i once found a model failing on a production dataset
# a single histogram of prediction probabilities showed a shifted distribution
# data drift due to a new data source
# this issue was visible immediately
# that's the power of visualization: it transforms confusion into clarity

# === 1.2 the visualization process: from raw data to insight
if False:
    #
    # creating effective visuals is an iterative process
    # it starts with curiosity and ends with understanding
    #
    # 1. ask the right question
    # what insight are you trying to uncover? the question drives the chart type
    # 2. inspect and prepare the data
    # handle missing values, clean formats, and explore types
    # 3. choose a visualization approach
    # pick the chart that aligns with your question
    # 4. iterate and interpret
    # plot, observe, refine, and draw conclusions
    #
    # let's walk through a practical workflow using Seaborn's built-in tips dataset:

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
    # in just a few lines, you move from raw data to a visual that reveals patterns
    # higher bills often mean higher tips, with difference between lunch and dinner

# === 1.3 choosing the right chart for the right data
if False:
    #
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
if False:
    #
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
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    sns.scatterplot(data=tips, x='total_bill', y='tip', hue='day', ax=ax)
    ax.set_title('tip vs total bill with outlier annotation')
    # annotate the highest total bill
    max_idx = tips['total_bill'].idxmax()
    row = tips.loc[max_idx]
    ax.annotate(f'highest bill: ${row.total_bill:.2f}',
                xy=(row.total_bill, row.tip),
                xytext=(row.total_bill+5, row.tip+1),
                arrowprops=dict(arrowstyle='->', lw=1.2) )
    plt.tight_layout()
    plt.show()
    # this single annotation turns a chart into a narrative
    # guiding attention where it matters most

# === 1.5 introduction to matplotlib and seaborn ecosystem
if False:
    #
    # matplotlib is python's foundational visualization library
    # it offers full control - down to pixel-level adjustment -
    # but can be verbose for complex plots
    #
    # seaborn builds on matplotlib, providing a high-level interface optimized for statistical visualization
    # it comes with attractive defaults, smart layouts, and build-in datasets that make experimentation easy
    #
    # here's how they complement each other:
    #
    # use seaborn for quick, insightful, and visually appealing plots
    # use matplotlib when you need fine control
    # custom figure layouts, annotations, or publication-ready adjustments
    # often, the best results come from combining both
    #
    # examples
    iris = sns.load_dataset('iris')
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', ax=ax)
    ax.set_title('iris sepal length vs width')
    ax.set_xlabel('sepal length (cm)')
    ax.set_ylabel('sepal width (cm)')
    plt.tight_layout()
    plt.show()
    # this approach - using seaborn for structure and matplotlib for refinement
    # will serve you throughout the book

# === summary
# visualization is the meeting point between analytical thinking and creative storytelling
# it transforms numbers into insights, making your data both understandable and memorable
# in this chapter, you learned how visualization fits into the data science workflow,
# how to move from raw data to meaningful visuals, and how to choose and design charts
# with clarity and purpose
# you also met the two tools - matplotlib and seaborn - that you'll master throughout this book

# === exercises
if True:
    # 1. load seaborn's tips dataset and create 2x2 subplot grid showing:
    # histogram of total_bill
    # boxplot of tip by day
    # scatter plot of total_bill vs top
    # bar plot of average tip by day
    tips = sns.load_dataset('tips')
    fig, ax = plt.subplots(2, 2, figsize=(9,7))
    sns.histplot(data=tips, x='total_bill', bins=20, kde=True, ax=ax[0,0])
    sns.boxplot(data=tips, x='day', y='tip', ax=ax[0,1])
    sns.scatterplot(data=tips, x='tip', y='total_bill', ax=ax[1,0])
    sns.barplot(data=tips, x='day', y='tip', ax=ax[1,1])
    ax[0,0].grid(ls=':')
    ax[0,1].grid(ls=':')
    ax[1,0].grid(ls=':')
    ax[1,1].grid(ls=':')
    plt.tight_layout()
    plt.show()
    plt.close()


    
