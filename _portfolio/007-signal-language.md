---
title: "Credit Card Fault detection"
layout: post
excerpt: "The overall goal of this project is to build a word recognizer for American Sign Language video sequences, demonstrating the power of probabalistic models."
tags: [Python, Machine learning, Udacity, Artificial Intelligence, Probabilistic Models]
header:
  teaser: asl.jpg
categories: portfolio
link:
share: true
comments: false
date: 24-01-2018
---
# Artificial Intelligence Engineer Nanodegree - Probabilistic Models
## Project: Sign Language Recognition System
- [Introduction](#intro)
- [Part 1 Feature Selection](#part1_tutorial)
    - [Tutorial](#part1_tutorial)
    - [Features Submission](#part1_submission)
    - [Features Unittest](#part1_test)
- [Part 2 Train the models](#part2_tutorial)
    - [Tutorial](#part2_tutorial)
    - [Model Selection Score Submission](#part2_submission)
    - [Model Score Unittest](#part2_test)
- [Part 3 Build a Recognizer](#part3_tutorial)
    - [Tutorial](#part3_tutorial)
    - [Recognizer Submission](#part3_submission)
    - [Recognizer Unittest](#part3_test)
- [Part 4 (OPTIONAL) Improve the WER with Language Models](#part4_info)

<a id='intro'></a>
## Introduction
The overall goal of this project is to build a word recognizer for American Sign Language video sequences, demonstrating the power of probabalistic models.  In particular, this project employs  [hidden Markov models (HMM's)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research (see the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php)).  In this video, the right-hand x and y locations are plotted as the speaker signs the sentence.
[![ASLR demo](http://www-i6.informatik.rwth-aachen.de/~dreuw/images/demosample.png)](https://drive.google.com/open?id=0B_5qGuFe-wbhUXRuVnNZVnMtam8)

The raw data, train, and test sets are pre-defined by Udacity.

<a id='part1_tutorial'></a>
## PART 1: Data

### Features Tutorial
##### Load the initial database
A data handler designed for this database is provided in the student codebase as the `AslDb` class in the `asl_data` module.  This handler creates the initial [pandas](http://pandas.pydata.org/pandas-docs/stable/) dataframe from the corpus of data included in the `data` directory as well as dictionaries suitable for extracting data in a format friendly to the [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/) library.  I'll use those to create models in Part 2.

To start, let's set up the initial database and select an example set of features for the training set.  At the end of Part 1, I will create additional feature sets for experimentation.

```python
import numpy as np
import pandas as pd
from asl_data import AslDb

asl = AslDb() # initializes the database
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>speaker</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
    </tr>
  </tbody>
</table>
</div>

```python
asl.df.ix[98,1]  # look at the data available for an individual frame
```

    left-x         149
    left-y         181
    right-x        170
    right-y        175
    nose-x         161
    nose-y          62
    speaker    woman-1
    Name: (98, 1), dtype: object


The frame represented by video 98, frame 1 is shown here:
![Video 98](http://www-i6.informatik.rwth-aachen.de/~dreuw/database/rwth-boston-104/overview/images/orig/098-start.jpg)

##### Feature selection for training the model
The objective of feature selection when training a model is to choose the most relevant variables while keeping the model as simple as possible, thus reducing training time.  We can use the raw features already provided or derive our own and add columns to the pandas dataframe `asl.df` for selection. As an example, in the next cell a feature named `'grnd-ry'` is added. This feature is the difference between the right-hand y value and the nose y value, which serves as the "ground" right y value.

```python
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df.head()  # the new feature 'grnd-ry' is now in the frames dictionary
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>speaker</th>
      <th>grnd-ry</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
    <tr>
      <th>3</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
    </tr>
  </tbody>
</table>
</div>



##### Trying it!


```python
from asl_utils import test_features_tryit
# Add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# test the code
test_features_tryit(asl)
```

    asl.df sample


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>speaker</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
    </tr>
  </tbody>
</table>
</div>


<font color=green>Correct!</font><br/>


```python
# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
[asl.df.ix[98,1][v] for v in features_ground]
```


    [9, 113, -12, 119]

##### Build the training set
Now that I have a feature list defined, I can pass that list to the `build_training` method to collect the features for all the words in the training set.  Each word in the training set has multiple examples from various videos.  Below we can see the unique words that have been loaded into the training set:


```python
training = asl.build_training(features_ground)
print("Training words: {}".format(training.words))
```

    Training words: ['JOHN', 'WRITE', 'HOMEWORK', 'IX-1P', 'SEE', 'YESTERDAY', 'IX', 'LOVE', 'MARY', 'CAN', 'GO', 'GO1', 'FUTURE', 'GO2', 'PARTY', 'FUTURE1', 'HIT', 'BLAME', 'FRED', 'FISH', 'WONT', 'EAT', 'BUT', 'CHICKEN', 'VEGETABLE', 'CHINA', 'PEOPLE', 'PREFER', 'BROCCOLI', 'LIKE', 'LEAVE', 'SAY', 'BUY', 'HOUSE', 'KNOW', 'CORN', 'CORN1', 'THINK', 'NOT', 'PAST', 'LIVE', 'CHICAGO', 'CAR', 'SHOULD', 'DECIDE', 'VISIT', 'MOVIE', 'WANT', 'SELL', 'TOMORROW', 'NEXT-WEEK', 'NEW-YORK', 'LAST-WEEK', 'WILL', 'FINISH', 'ANN', 'READ', 'BOOK', 'CHOCOLATE', 'FIND', 'SOMETHING-ONE', 'POSS', 'BROTHER', 'ARRIVE', 'HERE', 'GIVE', 'MAN', 'NEW', 'COAT', 'WOMAN', 'GIVE1', 'HAVE', 'FRANK', 'BREAK-DOWN', 'SEARCH-FOR', 'WHO', 'WHAT', 'LEG', 'FRIEND', 'CANDY', 'BLUE', 'SUE', 'BUY1', 'STOLEN', 'OLD', 'STUDENT', 'VIDEOTAPE', 'BORROW', 'MOTHER', 'POTATO', 'TELL', 'BILL', 'THROW', 'APPLE', 'NAME', 'SHOOT', 'SAY-1P', 'SELF', 'GROUP', 'JANA', 'TOY1', 'MANY', 'TOY', 'ALL', 'BOY', 'TEACHER', 'GIRL', 'BOX', 'GIVE2', 'GIVE3', 'GET', 'PUTASIDE']


The training data in `training` is an object of class `WordsData` defined in the `asl_data` module.  in addition to the `words` list, data can be accessed with the `get_all_sequences`, `get_all_Xlengths`, `get_word_sequences`, and `get_word_Xlengths` methods. We need the `get_word_Xlengths` method to train multiple sequences with the `hmmlearn` library.  In the following example, notice that there are two lists; the first is a concatenation of all the sequences(the X portion) and the second is a list of the sequence lengths(the Lengths portion).

```python
training.get_word_Xlengths('CHOCOLATE')
```

    (array([[-11,  48,   7, 120],
            [-11,  48,   8, 109],
            [ -8,  49,  11,  98],
            [ -7,  50,   7,  87],
            [ -4,  54,   7,  77],
            [ -4,  54,   6,  69],
            [ -4,  54,   6,  69],
            [-13,  52,   6,  69],
            [-13,  52,   6,  69],
            [ -8,  51,   6,  69],
            [ -8,  51,   6,  69],
            [ -8,  51,   6,  69],
            [ -8,  51,   6,  69],
            [ -8,  51,   6,  69],
            [-10,  59,   7,  71],
            [-15,  64,   9,  77],
            [-17,  75,  13,  81],
            [ -4,  48,  -4, 113],
            [ -2,  53,  -4, 113],
            [ -4,  55,   2,  98],
            [ -4,  58,   2,  98],
            [ -1,  59,   2,  89],
            [ -1,  59,  -1,  84],
            [ -1,  59,  -1,  84],
            [ -7,  63,  -1,  84],
            [ -7,  63,  -1,  84],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -7,  63,   3,  83],
            [ -4,  70,   3,  83],
            [ -4,  70,   3,  83],
            [ -2,  73,   5,  90],
            [ -3,  79,  -4,  96],
            [-15,  98,  13, 135],
            [ -6,  93,  12, 128],
            [ -2,  89,  14, 118],
            [  5,  90,  10, 108],
            [  4,  86,   7, 105],
            [  4,  86,   7, 105],
            [  4,  86,  13, 100],
            [ -3,  82,  14,  96],
            [ -3,  82,  14,  96],
            [  6,  89,  16, 100],
            [  6,  89,  16, 100],
            [  7,  85,  17, 111]], dtype=int64), [17, 20, 12])


###### More feature sets
So far we have a simple feature set that is enough to get started modeling.  However, we might get better results if we manipulate the raw values a bit more, so we will go ahead and set up some other options now for experimentation later.  For example, we could normalize each speaker's range of motion with grouped statistics using [Pandas stats](http://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-stats) functions and [pandas groupby](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html).  Below is an example for finding the means of all speaker subgroups.


```python
df_means = asl.df.groupby('speaker').mean()
df_means
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
    </tr>
    <tr>
      <th>speaker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>man-1</th>
      <td>206.248203</td>
      <td>218.679449</td>
      <td>155.464350</td>
      <td>150.371031</td>
      <td>175.031756</td>
      <td>61.642600</td>
      <td>88.728430</td>
      <td>-19.567406</td>
      <td>157.036848</td>
      <td>31.216447</td>
    </tr>
    <tr>
      <th>woman-1</th>
      <td>164.661438</td>
      <td>161.271242</td>
      <td>151.017865</td>
      <td>117.332462</td>
      <td>162.655120</td>
      <td>57.245098</td>
      <td>60.087364</td>
      <td>-11.637255</td>
      <td>104.026144</td>
      <td>2.006318</td>
    </tr>
    <tr>
      <th>woman-2</th>
      <td>183.214509</td>
      <td>176.527232</td>
      <td>156.866295</td>
      <td>119.835714</td>
      <td>170.318973</td>
      <td>58.022098</td>
      <td>61.813616</td>
      <td>-13.452679</td>
      <td>118.505134</td>
      <td>12.895536</td>
    </tr>
  </tbody>
</table>
</div>



To select a mean that matches by speaker, use the pandas [map](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) method:


```python
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>speaker</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
      <th>left-x-mean</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
    <tr>
      <th>3</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
    <tr>
      <th>4</th>
      <td>149</td>
      <td>181</td>
      <td>170</td>
      <td>175</td>
      <td>161</td>
      <td>62</td>
      <td>woman-1</td>
      <td>113</td>
      <td>9</td>
      <td>119</td>
      <td>-12</td>
      <td>164.661438</td>
    </tr>
  </tbody>
</table>
</div>


##### Trying it!

```python
from asl_utils import test_std_tryit
# Create a dataframe named `df_std` with standard deviations grouped by speaker
df_std = asl.df.groupby('speaker').std()
# test the code
test_std_tryit(df_std)
```

    df_std


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left-x</th>
      <th>left-y</th>
      <th>right-x</th>
      <th>right-y</th>
      <th>nose-x</th>
      <th>nose-y</th>
      <th>grnd-ry</th>
      <th>grnd-rx</th>
      <th>grnd-ly</th>
      <th>grnd-lx</th>
      <th>left-x-mean</th>
    </tr>
    <tr>
      <th>speaker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>man-1</th>
      <td>15.154425</td>
      <td>36.328485</td>
      <td>18.901917</td>
      <td>54.902340</td>
      <td>6.654573</td>
      <td>5.520045</td>
      <td>53.487999</td>
      <td>20.269032</td>
      <td>36.572749</td>
      <td>15.080360</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>woman-1</th>
      <td>17.573442</td>
      <td>26.594521</td>
      <td>16.459943</td>
      <td>34.667787</td>
      <td>3.549392</td>
      <td>3.538330</td>
      <td>33.972660</td>
      <td>16.764706</td>
      <td>27.117393</td>
      <td>17.328941</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>woman-2</th>
      <td>15.388711</td>
      <td>28.825025</td>
      <td>14.890288</td>
      <td>39.649111</td>
      <td>4.099760</td>
      <td>3.416167</td>
      <td>39.128572</td>
      <td>16.191324</td>
      <td>29.320655</td>
      <td>15.050938</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
<font color=green>Correct!</font><br/>


<a id='part1_submission'></a>

### Features Implementation Submission.


Implement four feature sets and answer the question that follows.
- normalized Cartesian coordinates
    - use *mean* and *standard deviation* statistics and the [standard score](https://en.wikipedia.org/wiki/Standard_score) equation to account for speakers with different heights and arm length

- polar coordinates
    - calculate polar coordinates with [Cartesian to polar equations](https://en.wikipedia.org/wiki/Polar_coordinate_system#Converting_between_polar_and_Cartesian_coordinates)
    - use the [np.arctan2](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.arctan2.html) function and *swap the x and y axes* to move the $0$ to $2\pi$ discontinuity to 12 o'clock instead of 3 o'clock;  in other words, the normal break in radians value from $0$ to $2\pi$ occurs directly to the left of the speaker's nose, which may be in the signing area and interfere with results.  By swapping the x and y axes, that discontinuity move to directly above the speaker's head, an area not generally used in signing.

- delta difference
    - as described in Thad's lecture, use the difference in values between one frame and the next frames as features
    - pandas [diff method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.diff.html) and [fillna method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html) will be helpful for this one

- custom features
    - These are your own design; combine techniques used above or come up with something else entirely. We look forward to seeing what you come up with!
    Some ideas to get you started:
        - normalize using a [feature scaling equation](https://en.wikipedia.org/wiki/Feature_scaling)
        - normalize the polar coordinates
        - adding additional deltas


```python
# Aadd features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd
#mean
asl.df['right-x-mean'] = asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean'] = asl.df['speaker'].map(df_means['right-y'])
asl.df['left-x-mean'] = asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean'] = asl.df['speaker'].map(df_means['left-y'])

means = ['right-x-mean', 'right-y-mean','left-x-mean','left-y-mean']
asl.df[means].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>right-x-mean</th>
      <th>right-y-mean</th>
      <th>left-x-mean</th>
      <th>left-y-mean</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>151.017865</td>
      <td>117.332462</td>
      <td>164.661438</td>
      <td>161.271242</td>
    </tr>
    <tr>
      <th>1</th>
      <td>151.017865</td>
      <td>117.332462</td>
      <td>164.661438</td>
      <td>161.271242</td>
    </tr>
    <tr>
      <th>2</th>
      <td>151.017865</td>
      <td>117.332462</td>
      <td>164.661438</td>
      <td>161.271242</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151.017865</td>
      <td>117.332462</td>
      <td>164.661438</td>
      <td>161.271242</td>
    </tr>
    <tr>
      <th>4</th>
      <td>151.017865</td>
      <td>117.332462</td>
      <td>164.661438</td>
      <td>161.271242</td>
    </tr>
  </tbody>
</table>
</div>


```python
#std
asl.df['right-x-std'] = asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std'] = asl.df['speaker'].map(df_std['right-y'])
asl.df['left-x-std'] = asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std'] = asl.df['speaker'].map(df_std['left-y'])

std_ = ['right-x-std', 'right-y-std','left-x-std','left-y-std']

asl.df[std_].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>right-x-std</th>
      <th>right-y-std</th>
      <th>left-x-std</th>
      <th>left-y-std</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>16.459943</td>
      <td>34.667787</td>
      <td>17.573442</td>
      <td>26.594521</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16.459943</td>
      <td>34.667787</td>
      <td>17.573442</td>
      <td>26.594521</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16.459943</td>
      <td>34.667787</td>
      <td>17.573442</td>
      <td>26.594521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.459943</td>
      <td>34.667787</td>
      <td>17.573442</td>
      <td>26.594521</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.459943</td>
      <td>34.667787</td>
      <td>17.573442</td>
      <td>26.594521</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Z-score scaling (X-Xmean)/Xstd
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

asl.df[features_norm].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>norm-rx</th>
      <th>norm-ry</th>
      <th>norm-lx</th>
      <th>norm-ly</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>1.153232</td>
      <td>1.663433</td>
      <td>-0.891199</td>
      <td>0.741835</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.153232</td>
      <td>1.663433</td>
      <td>-0.891199</td>
      <td>0.741835</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.153232</td>
      <td>1.663433</td>
      <td>-0.891199</td>
      <td>0.741835</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.153232</td>
      <td>1.663433</td>
      <td>-0.891199</td>
      <td>0.741835</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.153232</td>
      <td>1.663433</td>
      <td>-0.891199</td>
      <td>0.741835</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Add features for polar coordinate values where the **nose** is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
asl.df['polar-rr'] = np.sqrt(np.square(asl.df['grnd-rx']) + np.square(asl.df['grnd-ry']))
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'] = np.sqrt(np.square(asl.df['grnd-lx']) + np.square(asl.df['grnd-ly']))
asl.df['polar-ltheta'] =  np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
asl.df[features_polar].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>polar-rr</th>
      <th>polar-rtheta</th>
      <th>polar-lr</th>
      <th>polar-ltheta</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>113.35784</td>
      <td>0.079478</td>
      <td>119.603512</td>
      <td>-0.100501</td>
    </tr>
    <tr>
      <th>1</th>
      <td>113.35784</td>
      <td>0.079478</td>
      <td>119.603512</td>
      <td>-0.100501</td>
    </tr>
    <tr>
      <th>2</th>
      <td>113.35784</td>
      <td>0.079478</td>
      <td>119.603512</td>
      <td>-0.100501</td>
    </tr>
    <tr>
      <th>3</th>
      <td>113.35784</td>
      <td>0.079478</td>
      <td>119.603512</td>
      <td>-0.100501</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113.35784</td>
      <td>0.079478</td>
      <td>119.603512</td>
      <td>-0.100501</td>
    </tr>
  </tbody>
</table>
</div>


```python
# Add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'
asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
asl.df[features_delta].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>delta-rx</th>
      <th>delta-ry</th>
      <th>delta-lx</th>
      <th>delta-ly</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like
#from sklearn.preprocessing import MinMaxScaler
#asl.df.head()
#(x-min)/(max-min)

df_max = asl.df.groupby('speaker').max()
df_min = asl.df.groupby('speaker').min()

asl.df['rs-rx'] = asl.df['speaker'].map(df_min['right-x']) / (asl.df['speaker'].map(df_max['right-x']) - asl.df['speaker'].map(df_min['right-x']))
asl.df['rs-ry'] = asl.df['speaker'].map(df_min['right-y']) / (asl.df['speaker'].map(df_max['right-y']) - asl.df['speaker'].map(df_min['right-y']))
asl.df['rs-lx'] = asl.df['speaker'].map(df_min['left-x']) / (asl.df['speaker'].map(df_max['left-x']) - asl.df['speaker'].map(df_min['left-x']))
asl.df['rs-ly'] = asl.df['speaker'].map(df_min['left-y']) / (asl.df['speaker'].map(df_max['left-y']) - asl.df['speaker'].map(df_min['left-y']))


# Define a list named 'features_custom' for building the training set
features_custom = ['rs-rx', 'rs-ry', 'rs-lx', 'rs-ly']

asl.df[features_custom].head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>rs-rx</th>
      <th>rs-ry</th>
      <th>rs-lx</th>
      <th>rs-ly</th>
    </tr>
    <tr>
      <th>video</th>
      <th>frame</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">98</th>
      <th>0</th>
      <td>0.817391</td>
      <td>0.260563</td>
      <td>1.447917</td>
      <td>0.785047</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.817391</td>
      <td>0.260563</td>
      <td>1.447917</td>
      <td>0.785047</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.817391</td>
      <td>0.260563</td>
      <td>1.447917</td>
      <td>0.785047</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.817391</td>
      <td>0.260563</td>
      <td>1.447917</td>
      <td>0.785047</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.817391</td>
      <td>0.260563</td>
      <td>1.447917</td>
      <td>0.785047</td>
    </tr>
  </tbody>
</table>
</div>

For the custom feature I used **Rescaling** method. I did it because it is the most common method I use for normalising and I did want to contrast its performance towards **Standardization**. The performance is not so good as expected (WER average of 0.86).

<a id='part1_test'></a>
### Features Unit Testing
Run the following unit tests as a sanity check on the defined "ground", "norm", "polar", and 'delta"
feature sets.  The test simply looks for some valid values but is not exhaustive.  However, the project should not be submitted if these tests don't pass.

```python
import unittest
# import numpy as np

class TestFeatures(unittest.TestCase):

    def test_features_ground(self):
        sample = (asl.df.ix[98, 1][features_ground]).tolist()
        self.assertEqual(sample, [9, 113, -12, 119])

    def test_features_norm(self):
        sample = (asl.df.ix[98, 1][features_norm]).tolist()
        np.testing.assert_almost_equal(sample, [ 1.153,  1.663, -0.891,  0.742], 3)

    def test_features_polar(self):
        sample = (asl.df.ix[98,1][features_polar]).tolist()
        np.testing.assert_almost_equal(sample, [113.3578, 0.0794, 119.603, -0.1005], 3)

    def test_features_delta(self):
        sample = (asl.df.ix[98, 0][features_delta]).tolist()
        self.assertEqual(sample, [0, 0, 0, 0])
        sample = (asl.df.ix[98, 18][features_delta]).tolist()
        self.assertTrue(sample in [[-16, -5, -2, 4], [-14, -9, 0, 0]], "Sample value found was {}".format(sample))

suite = unittest.TestLoader().loadTestsFromModule(TestFeatures())
unittest.TextTestRunner().run(suite)
```

    ....
    ----------------------------------------------------------------------
    Ran 4 tests in 0.016s

    OK

    <unittest.runner.TextTestResult run=4 errors=0 failures=0>

<a id='part2_tutorial'></a>
## PART 2: Model Selection
### Model Selection Tutorial
The objective of Model Selection is to tune the number of states for each word HMM prior to testing on unseen data.  In this section you will explore three methods:
- Log likelihood using cross-validation folds (CV)
- Bayesian Information Criterion (BIC)
- Discriminative Information Criterion (DIC)

##### Train a single word
Now that we have built a training set with sequence data, we can "train" models for each word.  As a simple starting example, we train a single word using Gaussian hidden Markov models (HMM).   By using the `fit` method during training, the [Baum-Welch Expectation-Maximization](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) (EM) algorithm is invoked iteratively to find the best estimate for the model *for the number of hidden states specified* from a group of sample seequences. For this example, we *assume* the correct number of hidden states is 3, but that is just a guess.  How do we know what the "best" number of states for training is?  We will need to find some model selection technique to choose the best parameter.


```python
import warnings
from hmmlearn.hmm import GaussianHMM

def train_a_word(word, num_hidden_states, features):

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))
```

    Number of states trained in model for BOOK is 3
    logL = -2331.113812743319


The HMM model has been trained and information can be pulled from the model, including means and variances for each feature and hidden state.  The [log likelihood](http://math.stackexchange.com/questions/892832/why-we-consider-log-likelihood-instead-of-likelihood-in-gaussian-distribution) for any individual sample or group of samples can also be calculated with the `score` method.


```python
def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()

show_model_stats(demoword, model)
```

    Number of states trained in model for BOOK is 3
    hidden state #0
    mean =  [ -3.46504869  50.66686933  14.02391587  52.04731066]
    variance =  [ 49.12346305  43.04799144  39.35109609  47.24195772]

    hidden state #1
    mean =  [ -11.45300909   94.109178     19.03512475  102.2030162 ]
    variance =  [  77.403668    203.35441965   26.68898447  156.12444034]

    hidden state #2
    mean =  [ -1.12415027  69.44164191  17.02866283  77.7231196 ]
    variance =  [ 19.70434594  16.83041492  30.51552305  11.03678246]


##### Trying it!
Experiment by changing the feature set, word, and/or num_hidden_states values in the next cell to see changes in values.


```python
my_testword = 'CHOCOLATE'
model, logL = train_a_word(my_testword, 3, features_ground) # Experiment here with different parameters
show_model_stats(my_testword, model)
print("logL = {}".format(logL))
```

    Number of states trained in model for CHOCOLATE is 3
    hidden state #0
    mean =  [ -9.30211403  55.32333876   6.92259936  71.24057775]
    variance =  [ 16.16920957  46.50917372   3.81388185  15.79446427]

    hidden state #1
    mean =  [   0.58333333   87.91666667   12.75        108.5       ]
    variance =  [  39.41055556   18.74388889    9.855       144.4175    ]

    hidden state #2
    mean =  [ -5.40587658  60.1652424    2.32479599  91.3095432 ]
    variance =  [   7.95073876   64.13103127   13.68077479  129.5912395 ]

    logL = -601.3291470028621


##### Visualize the hidden states
We can plot the means and variances for each state and feature.  I have to try varying the number of states trained for the HMM model and examine the variances.  Are there some models that are "better" than others?  How can you tell?  We would like to hear what you think in the classroom online.

```python
%matplotlib inline
```

```python
import math
from matplotlib import (cm, pyplot as plt, mlab)

def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()

visualize(my_testword, model)
```

![png](/AI-Signal_lang/asl_recognizer_1.png)


![png](/AI-Signal_lang/asl_recognizer_2.png)


![png](/AI-Signal_lang/asl_recognizer_3.png)


![png](/AI-Signal_lang/asl_recognizer_4.png)


#####  ModelSelector class
Review the `ModelSelector` class from the codebase found in the `my_model_selectors.py` module.  It is designed to be a strategy pattern for choosing different model selectors.  For the project submission in this section, subclass `SelectorModel` to implement the following model selectors.  In other words, you will write your own classes/functions in the `my_model_selectors.py` module and run them from this notebook:

- `SelectorCV `:  Log likelihood with CV
- `SelectorBIC`: BIC
- `SelectorDIC`: DIC

You will train each word in the training set with a range of values for the number of hidden states, and then score these alternatives with the model selector, choosing the "best" according to each strategy. The simple case of training with a constant value for `n_components` can be called using the provided `SelectorConstant` subclass as follow:


```python
from my_model_selectors import SelectorConstant

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
word = 'VEGETABLE' # Experiment here with different words
model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3).select()
print("Number of states trained in model for {} is {}".format(word, model.n_components))
```

    Number of states trained in model for VEGETABLE is 3


##### Cross-validation folds
If we simply score the model with the Log Likelihood calculated from the feature sequences it has been trained on, we should expect that more complex models will have higher likelihoods. However, that doesn't tell us which would have a better likelihood score on unseen data.  The model will likely be overfit as complexity is added.  To estimate which topology model is better using only the training data, we can compare scores using cross-validation.  One technique for cross-validation is to break the training set into "folds" and rotate which fold is left out of training.  The "left out" fold scored.  This gives us a proxy method of finding the best model to use on "unseen data". In the following example, a set of word sequences is broken into three folds using the [scikit-learn Kfold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) class object. When you implement `SelectorCV`, you will use this technique.


```python
from sklearn.model_selection import KFold

training = asl.build_training(features_ground) # Experiment here with different feature sets
word = 'VEGETABLE' # Experiment here with different words
word_sequences = training.get_word_sequences(word)
split_method = KFold()
for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
```

    Train fold indices:[2 3 4 5] Test fold indices:[0 1]
    Train fold indices:[0 1 4 5] Test fold indices:[2 3]
    Train fold indices:[0 1 2 3] Test fold indices:[4 5]


**Tip:** In order to run `hmmlearn` training using the X,lengths tuples on the new folds, subsets must be combined based on the indices given for the folds.  A helper utility has been provided in the `asl_utils` module named `combine_sequences` for this purpose.

##### Scoring models with other criterion
Scoring model topologies with **BIC** balances fit and complexity within the training set for each word.  In the BIC equation, a penalty term penalizes complexity to avoid overfitting, so that it is not necessary to also use cross-validation in the selection process.  There are a number of references on the internet for this criterion.  These [slides](http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf) include a formula you may find helpful for your implementation.

The advantages of scoring model topologies with **DIC** over BIC are presented by Alain Biem in this [reference](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf) (also found [here](https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf)).  DIC scores the discriminant ability of a training set for one word against competing words.  Instead of a penalty term for complexity, it provides a penalty if model liklihoods for non-matching words are too similar to model likelihoods for the correct word in the word set.

<a id='part2_submission'></a>
### Model Selection Implementation Submission
Implement `SelectorCV`, `SelectorBIC`, and `SelectorDIC` classes in the `my_model_selectors.py` module.  Run the selectors on the following five words. Then answer the questions about your results.

**Tip:** The `hmmlearn` library may not be able to train or score all models.  Implement try/except contructs as necessary to eliminate non-viable models from consideration.


```python
words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit
```


```python
# autoreload for automatically reloading changes made in my_model_selectors and my_recognizer
%load_ext autoreload
%autoreload 2
```


```python
# Implement SelectorCV in my_model_selector.py
from my_model_selectors import SelectorCV

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorCV(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
```

    Training complete for FISH with 3 states with time 0.03563961881471995 seconds
    Training complete for BOOK with 3 states with time 0.09627137250351978 seconds
    Training complete for VEGETABLE with 3 states with time 0.027836075157135154 seconds
    Training complete for FUTURE with 3 states with time 0.06697517297698921 seconds
    Training complete for JOHN with 3 states with time 1.0956872335061059 seconds



```python
# Implement SelectorBIC in module my_model_selectors.py
from my_model_selectors import SelectorBIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorBIC(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
```

    Training complete for FISH with 2 states with time 0.48659670981143677 seconds
    Training complete for BOOK with 2 states with time 4.312977569330247 seconds
    Training complete for VEGETABLE with 2 states with time 1.2335360044267851 seconds
    Training complete for FUTURE with 2 states with time 3.378847831568052 seconds
    Training complete for JOHN with 2 states with time 28.793436580310185 seconds



```python
# Implement SelectorDIC in module my_model_selectors.py
from my_model_selectors import SelectorDIC

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    model = SelectorDIC(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))
```

    Training complete for FISH with 2 states with time 0.4920735149415023 seconds
    Training complete for BOOK with 2 states with time 4.209543911946298 seconds
    Training complete for VEGETABLE with 2 states with time 1.2439862466280402 seconds
    Training complete for FUTURE with 2 states with time 3.5207619243118273 seconds
    Training complete for JOHN with 2 states with time 30.458282256499047 seconds


**Question 2:**  Compare and contrast the possible advantages and disadvantages of the various model selectors implemented.

**Answer 2:**
The following table gathers the output of the different model selectors as x-y, where x represents the number of hidden states and y the time of computation expressed in seconds.

Word  |Cross-validation   | BIC   | DIC
--|---|---|--
FISH | 3 - 0.035639|2 - 0.486596|2 - 0.49207
BOOK | 3 - 0.096271| 2 - 4.3129|2 - 4.2095
VEGETABLE | 3 - 0.0278360| 2 - 1.23353|2 - 1.24398
FUTURE | 3 - 0.066975| 2 - 3.3788|2 - 3.52076
JOHN | 3 - 1.09568| 2 - 28.7934|2 - 30.4582

Cross-validation is, by far, the quickest model using only three states (1 above the minimum parameter). The advantage of using Cross-validation is that testing the model on "unseen" folders simulates the model for real "unseen" data. So scores are an approximation for new data.  The problem with CV is that you need an acceptable amount of data that makes the folding worth.
Regarding BIC, it penalises the more complex models, so it is its way to avoid overfitting. In contrast, the time computation is much more than comparing to CV.
DIC, penalises differently. It does it penalising models that give a similar log likelihood for non-matching word. Experiments show that DIC is better than BIC for achieving a high-performing system, although sometimes it means adding computing complexity.


<a id='part2_test'></a>
### Model Selector Unit Testing
Run the following unit tests as a sanity check on the implemented model selectors.  The test simply looks for valid interfaces  but is not exhaustive. However, the project should not be submitted if these tests don't pass.


```python
from asl_test_model_selectors import TestSelectors
suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)
```
    ....
    ----------------------------------------------------------------------
    Ran 4 tests in 33.049s

    OK

    <unittest.runner.TextTestResult run=4 errors=0 failures=0>

<a id='part3_tutorial'></a>
## PART 3: Recognizer
The objective of this section is to "put it all together".  Using the four feature sets created and the three model selectors, you will experiment with the models and present your results.  Instead of training only five specific words as in the previous section, train the entire set with a feature set and model selector strategy.
### Recognizer Tutorial
##### Train the full training set
The following example trains the entire set with the example `features_ground` and `SelectorConstant` features and model selector.  Use this pattern for you experimentation and final submission cells.

```python
from my_model_selectors import SelectorConstant

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

models = train_all_words(features_ground, SelectorConstant)
print("Number of word models returned = {}".format(len(models)))
```
    Number of word models returned = 112

##### Load the test set
The `build_test` method in `ASLdb` is similar to the `build_training` method already presented, but there are a few differences:
- the object is type `SinglesData`
- the internal dictionary keys are the index of the test word rather than the word itself
- the getter methods are `get_all_sequences`, `get_all_Xlengths`, `get_item_sequences` and `get_item_Xlengths`

```python
test_set = asl.build_test(features_ground)
print("Number of test set items: {}".format(test_set.num_items))
print("Number of test set sentences: {}".format(len(test_set.sentences_index)))
```

    Number of test set items: 178
    Number of test set sentences: 40


<a id='part3_submission'></a>
### Recognizer Implementation Submission
For the final project submission, students must implement a recognizer following guidance in the `my_recognizer.py` module.  Experiment with the four feature sets and the three model selection methods (that's 12 possible combinations). You can add and remove cells for experimentation or run the recognizers locally in some other way during your experiments, but retain the results for your discussion.  For submission, you will provide code cells of **only three** interesting combinations for your discussion (see questions below). At least one of these should produce a word error rate of less than 60%, i.e. WER < 0.60 .

**Tip:** The hmmlearn library may not be able to train or score all models.  Implement try/except contructs as necessary to eliminate non-viable models from consideration.


```python
# Implement the recognize method in my_recognizer
from my_recognizer import recognize
from asl_utils import show_errors
```
```python
# Choose a feature set and model selector

#features_ground, features_norm, feautures_polar, features_delta, features_custom
features = features_delta# change as needed

# SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV
model_selector = SelectorConstant # change as needed

# Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)
```
    **** WER = 0.6404494382022472
    Total correct: 64 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
        2: JOHN *JOHN HOMEWORK                                           JOHN WRITE HOMEWORK
        7: JOHN *HAVE *GIVE1 *TEACHER                                    JOHN CAN GO CAN
       12: JOHN CAN *GO1 CAN                                             JOHN CAN GO CAN
       21: *MARY *MARY *JOHN *MARY *CAR *GO *FUTURE *MARY                JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN *MARY *JOHN IX *MARY                                     JOHN LIKE IX IX IX
       28: JOHN *MARY *MARY IX IX                                        JOHN LIKE IX IX IX
       30: JOHN *MARY *JOHN *JOHN IX                                     JOHN LIKE IX IX IX
       36: MARY *JOHN *JOHN IX *MARY *MARY                               MARY VEGETABLE KNOW IX LIKE CORN1
       40: *MARY IX *MARY MARY *MARY                                     JOHN IX THINK MARY LOVE
       43: JOHN *JOHN *FINISH HOUSE                                      JOHN MUST BUY HOUSE
       50: *JOHN JOHN BUY CAR *MARY                                      FUTURE JOHN BUY CAR SHOULD
       54: JOHN *MARY *MARY BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
       57: JOHN *JOHN *IX *JOHN                                          JOHN DECIDE VISIT MARY
       67: JOHN *JOHN *JOHN BUY HOUSE                                    JOHN FUTURE NOT BUY HOUSE
       71: JOHN *JOHN VISIT MARY                                         JOHN WILL VISIT MARY
       74: JOHN *JOHN *MARY MARY                                         JOHN NOT VISIT MARY
       77: *JOHN BLAME MARY                                              ANN BLAME MARY
       84: *JOHN *GO *IX *WHAT                                           IX-1P FIND SOMETHING-ONE BOOK
       89: *GIVE1 *JOHN *IX *JOHN IX *WHAT *HOUSE                        JOHN IX GIVE MAN IX NEW COAT
       90: *MARY *JOHN *JOHN *IX *IX *MARY                               JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       92: JOHN *MARY *JOHN *JOHN WOMAN *ARRIVE                          JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
      100: *JOHN NEW *WHAT BREAK-DOWN                                    POSS NEW CAR BREAK-DOWN
      105: JOHN *MARY                                                    JOHN LEG
      107: JOHN POSS FRIEND *LOVE *MARY                                  JOHN POSS FRIEND HAVE CANDY
      108: *JOHN ARRIVE                                                  WOMAN ARRIVE
      113: *JOHN CAR *MARY *MARY *GIVE1                                  IX CAR BLUE SUE BUY
      119: *JOHN *BUY1 IX CAR *IX                                        SUE BUY IX CAR BLUE
      122: JOHN *VISIT *YESTERDAY                                        JOHN READ BOOK
      139: JOHN *BUY1 WHAT *MARY *ARRIVE                                 JOHN BUY WHAT YESTERDAY BOOK
      142: JOHN BUY *MARY *MARY *YESTERDAY                               JOHN BUY YESTERDAY WHAT BOOK
      158: *BOY *WHO *MARY                                               LOVE JOHN WHO
      167: *MARY *MARY *IX *ARRIVE *WHAT                                 JOHN IX SAY LOVE MARY
      171: JOHN *JOHN BLAME                                              JOHN MARY BLAME
      174: *GIVE1 *MARY GIVE1 *MARY *FINISH                              PEOPLE GROUP GIVE1 JANA TOY
      181: JOHN *GIVE1                                                   JOHN ARRIVE
      184: *IX *WHO *GIVE1 *HAVE *MARY                                   ALL BOY GIVE TEACHER APPLE
      189: JOHN *IX *MARY *VISIT                                         JOHN GIVE GIRL BOX
      193: JOHN *IX *IX BOX                                              JOHN GIVE GIRL BOX
      199: *JOHN *ARRIVE *MARY                                           LIKE CHOCOLATE WHO
      201: JOHN *MARY MARY *LIKE *VISIT HOUSE                            JOHN TELL MARY IX-1P BUY HOUSE

```python
# Choose a feature set and model selector
#features_ground, features_norm, feautures_polar, features_delta, features_custom
features =  features_delta
# SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV
model_selector = SelectorBIC
# Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)
```

    **** WER = 0.6067415730337079
    Total correct: 70 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
        2: JOHN *IX HOMEWORK                                             JOHN WRITE HOMEWORK
        7: JOHN *HAVE *GIVE1 *TEACHER                                    JOHN CAN GO CAN
       12: JOHN CAN *GO1 CAN                                             JOHN CAN GO CAN
       21: JOHN *MARY WONT *BOOK *BUY *VISIT *MARY *BROCCOLI             JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN *MARY *MARY IX *MARY                                     JOHN LIKE IX IX IX
       28: JOHN *MARY *JOHN IX IX                                        JOHN LIKE IX IX IX
       30: JOHN *IX *MARY *JOHN IX                                       JOHN LIKE IX IX IX
       36: MARY *JOHN *JOHN IX *MARY *JOHN                               MARY VEGETABLE KNOW IX LIKE CORN1
       40: JOHN IX *CHINA MARY *MARY                                     JOHN IX THINK MARY LOVE
       43: JOHN *MARY *FINISH HOUSE                                      JOHN MUST BUY HOUSE
       50: *JOHN *MARY BUY CAR *MARY                                     FUTURE JOHN BUY CAR SHOULD
       54: JOHN *JOHN *JOHN BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
       57: JOHN *MARY *IX *JOHN                                          JOHN DECIDE VISIT MARY
       67: JOHN *JOHN *MARY *BUT HOUSE                                   JOHN FUTURE NOT BUY HOUSE
       71: JOHN *JOHN *FINISH *BOOK                                      JOHN WILL VISIT MARY
       74: JOHN *JOHN *IX *IX                                            JOHN NOT VISIT MARY
       77: *JOHN BLAME MARY                                              ANN BLAME MARY
       84: *JOHN *CAR *IX *HERE                                          IX-1P FIND SOMETHING-ONE BOOK
       89: JOHN *JOHN *IX *JOHN IX *WHAT *HOUSE                          JOHN IX GIVE MAN IX NEW COAT
       90: *MARY *JOHN *JOHN *IX *IX *MARY                               JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       92: JOHN *IX *JOHN *JOHN WOMAN BOOK                               JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
      100: *JOHN NEW *WHAT BREAK-DOWN                                    POSS NEW CAR BREAK-DOWN
      105: JOHN *JOHN                                                    JOHN LEG
      107: JOHN POSS FRIEND *FUTURE *MARY                                JOHN POSS FRIEND HAVE CANDY
      108: *JOHN *BOOK                                                   WOMAN ARRIVE
      113: *JOHN CAR *BROCCOLI *MARY *BUY1                               IX CAR BLUE SUE BUY
      119: *JOHN *GIVE1 IX CAR *MARY                                     SUE BUY IX CAR BLUE
      122: JOHN *LOVE *HERE                                              JOHN READ BOOK
      139: JOHN *BUY1 WHAT *JOHN BOOK                                    JOHN BUY WHAT YESTERDAY BOOK
      142: JOHN BUY *IX *MARY BOOK                                       JOHN BUY YESTERDAY WHAT BOOK
      158: LOVE *WHO WHO                                                 LOVE JOHN WHO
      167: JOHN IX *IX *BOOK *BOOK                                       JOHN IX SAY LOVE MARY
      171: JOHN *JOHN BLAME                                              JOHN MARY BLAME
      174: *GIVE1 *MARY GIVE1 *MARY *FINISH                              PEOPLE GROUP GIVE1 JANA TOY
      181: JOHN *GIVE1                                                   JOHN ARRIVE
      184: *IX *WHAT *GIVE1 TEACHER *MARY                                ALL BOY GIVE TEACHER APPLE
      189: JOHN *IX *BROCCOLI *WHAT                                      JOHN GIVE GIRL BOX
      193: JOHN *IX *IX BOX                                              JOHN GIVE GIRL BOX
      199: *JOHN *BOOK *BROCCOLI                                         LIKE CHOCOLATE WHO
      201: JOHN *IX *BROCCOLI *WOMAN BUY HOUSE                           JOHN TELL MARY IX-1P BUY HOUSE



```python
# Choose a feature set and model selector
#features_ground, features_norm, feautures_polar, features_delta, features_custom
features = features_delta
# SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV
model_selector = SelectorDIC
# Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)
```


    **** WER = 0.6067415730337079
    Total correct: 70 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
        2: JOHN *IX HOMEWORK                                             JOHN WRITE HOMEWORK
        7: JOHN *HAVE *GIVE1 *TEACHER                                    JOHN CAN GO CAN
       12: JOHN CAN *GO1 CAN                                             JOHN CAN GO CAN
       21: JOHN *MARY WONT *BOOK *BUY *VISIT *MARY *BROCCOLI             JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN *MARY *MARY IX *MARY                                     JOHN LIKE IX IX IX
       28: JOHN *MARY *JOHN IX IX                                        JOHN LIKE IX IX IX
       30: JOHN *IX *MARY *JOHN IX                                       JOHN LIKE IX IX IX
       36: MARY *JOHN *JOHN IX *MARY *JOHN                               MARY VEGETABLE KNOW IX LIKE CORN1
       40: JOHN IX *CHINA MARY *MARY                                     JOHN IX THINK MARY LOVE
       43: JOHN *MARY *FINISH HOUSE                                      JOHN MUST BUY HOUSE
       50: *JOHN *MARY BUY CAR *MARY                                     FUTURE JOHN BUY CAR SHOULD
       54: JOHN *JOHN *JOHN BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
       57: JOHN *MARY *IX *JOHN                                          JOHN DECIDE VISIT MARY
       67: JOHN *JOHN *MARY *BUT HOUSE                                   JOHN FUTURE NOT BUY HOUSE
       71: JOHN *JOHN *FINISH *BOOK                                      JOHN WILL VISIT MARY
       74: JOHN *JOHN *IX *IX                                            JOHN NOT VISIT MARY
       77: *JOHN BLAME MARY                                              ANN BLAME MARY
       84: *JOHN *CAR *IX *HERE                                          IX-1P FIND SOMETHING-ONE BOOK
       89: JOHN *JOHN *IX *JOHN IX *WHAT *HOUSE                          JOHN IX GIVE MAN IX NEW COAT
       90: *MARY *JOHN *JOHN *IX *IX *MARY                               JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       92: JOHN *IX *JOHN *JOHN WOMAN BOOK                               JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
      100: *JOHN NEW *WHAT BREAK-DOWN                                    POSS NEW CAR BREAK-DOWN
      105: JOHN *JOHN                                                    JOHN LEG
      107: JOHN POSS FRIEND *FUTURE *MARY                                JOHN POSS FRIEND HAVE CANDY
      108: *JOHN *BOOK                                                   WOMAN ARRIVE
      113: *JOHN CAR *BROCCOLI *MARY *BUY1                               IX CAR BLUE SUE BUY
      119: *JOHN *GIVE1 IX CAR *MARY                                     SUE BUY IX CAR BLUE
      122: JOHN *LOVE *HERE                                              JOHN READ BOOK
      139: JOHN *BUY1 WHAT *JOHN BOOK                                    JOHN BUY WHAT YESTERDAY BOOK
      142: JOHN BUY *IX *MARY BOOK                                       JOHN BUY YESTERDAY WHAT BOOK
      158: LOVE *WHO WHO                                                 LOVE JOHN WHO
      167: JOHN IX *IX *BOOK *BOOK                                       JOHN IX SAY LOVE MARY
      171: JOHN *JOHN BLAME                                              JOHN MARY BLAME
      174: *GIVE1 *MARY GIVE1 *MARY *FINISH                              PEOPLE GROUP GIVE1 JANA TOY
      181: JOHN *GIVE1                                                   JOHN ARRIVE
      184: *IX *WHAT *GIVE1 TEACHER *MARY                                ALL BOY GIVE TEACHER APPLE
      189: JOHN *IX *BROCCOLI *WHAT                                      JOHN GIVE GIRL BOX
      193: JOHN *IX *IX BOX                                              JOHN GIVE GIRL BOX
      199: *JOHN *BOOK *BROCCOLI                                         LIKE CHOCOLATE WHO
      201: JOHN *IX *BROCCOLI *WOMAN BUY HOUSE                           JOHN TELL MARY IX-1P BUY HOUSE



```python
# Choose a feature set and model selector

#features_ground, features_norm, feautures_polar, features_delta, features_custom
features = features_delta # change as needed

# SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV
model_selector = SelectorCV # change as needed

# Recognize the test set and display the result with the show_errors method
models = train_all_words(features, model_selector)
test_set = asl.build_test(features)
probabilities, guesses = recognize(models, test_set)
show_errors(guesses, test_set)
```


    **** WER = 0.6404494382022472
    Total correct: 64 out of 178
    Video  Recognized                                                    Correct
    =====================================================================================================
        2: JOHN *JOHN HOMEWORK                                           JOHN WRITE HOMEWORK
        7: JOHN *HAVE *GIVE1 *TEACHER                                    JOHN CAN GO CAN
       12: JOHN CAN *GO1 CAN                                             JOHN CAN GO CAN
       21: *MARY *MARY *JOHN *MARY *CAR *GO *FUTURE *MARY                JOHN FISH WONT EAT BUT CAN EAT CHICKEN
       25: JOHN *MARY *JOHN IX *MARY                                     JOHN LIKE IX IX IX
       28: JOHN *MARY *MARY IX IX                                        JOHN LIKE IX IX IX
       30: JOHN *MARY *JOHN *JOHN IX                                     JOHN LIKE IX IX IX
       36: MARY *JOHN *JOHN IX *MARY *MARY                               MARY VEGETABLE KNOW IX LIKE CORN1
       40: *MARY IX *MARY MARY *MARY                                     JOHN IX THINK MARY LOVE
       43: JOHN *JOHN *FINISH HOUSE                                      JOHN MUST BUY HOUSE
       50: *JOHN JOHN BUY CAR *MARY                                      FUTURE JOHN BUY CAR SHOULD
       54: JOHN *MARY *MARY BUY HOUSE                                    JOHN SHOULD NOT BUY HOUSE
       57: JOHN *JOHN *IX *JOHN                                          JOHN DECIDE VISIT MARY
       67: JOHN *JOHN *JOHN BUY HOUSE                                    JOHN FUTURE NOT BUY HOUSE
       71: JOHN *JOHN VISIT MARY                                         JOHN WILL VISIT MARY
       74: JOHN *JOHN *MARY MARY                                         JOHN NOT VISIT MARY
       77: *JOHN BLAME MARY                                              ANN BLAME MARY
       84: *JOHN *GO *IX *WHAT                                           IX-1P FIND SOMETHING-ONE BOOK
       89: *GIVE1 *JOHN *IX *JOHN IX *WHAT *HOUSE                        JOHN IX GIVE MAN IX NEW COAT
       90: *MARY *JOHN *JOHN *IX *IX *MARY                               JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
       92: JOHN *MARY *JOHN *JOHN WOMAN *ARRIVE                          JOHN GIVE IX SOMETHING-ONE WOMAN BOOK
      100: *JOHN NEW *WHAT BREAK-DOWN                                    POSS NEW CAR BREAK-DOWN
      105: JOHN *MARY                                                    JOHN LEG
      107: JOHN POSS FRIEND *LOVE *MARY                                  JOHN POSS FRIEND HAVE CANDY
      108: *JOHN ARRIVE                                                  WOMAN ARRIVE
      113: *JOHN CAR *MARY *MARY *GIVE1                                  IX CAR BLUE SUE BUY
      119: *JOHN *BUY1 IX CAR *IX                                        SUE BUY IX CAR BLUE
      122: JOHN *VISIT *YESTERDAY                                        JOHN READ BOOK
      139: JOHN *BUY1 WHAT *MARY *ARRIVE                                 JOHN BUY WHAT YESTERDAY BOOK
      142: JOHN BUY *MARY *MARY *YESTERDAY                               JOHN BUY YESTERDAY WHAT BOOK
      158: *BOY *WHO *MARY                                               LOVE JOHN WHO
      167: *MARY *MARY *IX *ARRIVE *WHAT                                 JOHN IX SAY LOVE MARY
      171: JOHN *JOHN BLAME                                              JOHN MARY BLAME
      174: *GIVE1 *MARY GIVE1 *MARY *FINISH                              PEOPLE GROUP GIVE1 JANA TOY
      181: JOHN *GIVE1                                                   JOHN ARRIVE
      184: *IX *WHO *GIVE1 *HAVE *MARY                                   ALL BOY GIVE TEACHER APPLE
      189: JOHN *IX *MARY *VISIT                                         JOHN GIVE GIRL BOX
      193: JOHN *IX *IX BOX                                              JOHN GIVE GIRL BOX
      199: *JOHN *ARRIVE *MARY                                           LIKE CHOCOLATE WHO
      201: JOHN *MARY MARY *LIKE *VISIT HOUSE                            JOHN TELL MARY IX-1P BUY HOUSE


Although in the script only three combinations are shown, I did run all possible combinations and summarize the output in the following table:


  |features_ground    |features_norm   | feautures_polar  |  features_delta  |   feautures_custom
--|---|---|---|---|--
**SelectorConstant**  | WER = 0.6685393258426966 (Total correct: 59 out of 178)	|	WER = 0.6235955056179775 (Total correct: 67 out of 178)	|	WER = 0.6179775280898876 (Total correct: 68 out of 178)	|	WER = 0.6404494382022472 (Total correct: 64 out of 178)	|	WER = 0.8314606741573034 (Total correct: 30 out of 178)
**SelectorBIC**  |WER = 0.6348314606741573 (Total correct: 65 out of 178)	|	WER = 0.6629213483146067 (Total correct: 60 out of 178)	|	WER = 0.6404494382022472 (Total correct: 64 out of 178)	|	WER = 0.6067415730337079 (Total correct: 70 out of 178)	|	WER = 0.898876404494382 (Total correct: 18 out of 178)
 **SelectorDIC**  |WER = 0.6348314606741573 (Total correct: 65 out of 178)	|	WER = 0.6629213483146067 (Total correct: 60 out of 178)	|	WER = 0.6404494382022472 (Total correct: 64 out of 178)	|	WER = 0.6067415730337079 (Total correct: 70 out of 178)	|	WER = 0.898876404494382 (Total correct: 18 out of 178)
  **SelectorCV**  |WER = 0.6685393258426966 (Total correct: 59 out of 178)	|	WER = 0.6235955056179775 (Total correct: 67 out of 178)	|	WER = 0.6179775280898876 (Total correct: 68 out of 178)	|	WER = 0.6404494382022472 (Total correct: 64 out of 178)	|	 WER = 0.8314606741573034 (Total correct: 30 out of 178)

In my running, I got less Word Error Rates with the combined features delta and selectorsBIC and DIC.
They are both tied concerning performance.
Feature_delta is expreseed as the difference in values between one frame and the next frames as features, so it is focus on transition. Probably because of that it is more acuurate to detect the next word. In addition, Discriminative methods such as DIC are less likely to have generated data belonging to competing classification categories.

To improve WER we could try adding more features, removing them or creating new feature combinations. Also, we can add more context to the words. A thing that can add more complexity thus time computation, but will help to better performance

<a id='part3_test'></a>
### Recognizer Unit Tests
Run the following unit tests as a sanity check on the defined recognizer.  The test simply looks for some valid values but is not exhaustive. However, the project should not be submitted if these tests don't pass.

```python
from asl_test_recognizer import TestRecognize
suite = unittest.TestLoader().loadTestsFromModule(TestRecognize())
unittest.TextTestRunner().run(suite)
```

    ..
    ----------------------------------------------------------------------
    Ran 2 tests in 59.307s

    OK

    <unittest.runner.TextTestResult run=2 errors=0 failures=0>



<a id='part4_info'></a>
## PART 4: Improve the WER with Language Models
We've squeezed just about as much as we can out of the model and still only get about 50% of the words right! Surely we can do better than that.  Probability to the rescue again in the form of [statistical language models (SLM)](https://en.wikipedia.org/wiki/Language_model).  The basic idea is that each word has some probability of occurrence within the set, and some probability that it is adjacent to specific other words. We can use that additional information to make better choices.

##### Additional reading and resources
- [Introduction to N-grams (Stanford Jurafsky slides)](https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf)
- [Speech Recognition Techniques for a Sign Language Recognition System, Philippe Dreuw et al](https://www-i6.informatik.rwth-aachen.de/publications/download/154/Dreuw--2007.pdf) see the improved results of applying LM on *this* data!
- [SLM data for *this* ASL dataset](ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-boston-104/lm/)

##### Optional challenge
The recognizer implemented in Part 3 is equivalent to a "0-gram" SLM.  Improve the WER with the SLM data provided with the data set in the link above using "1-gram", "2-gram", and/or "3-gram" statistics. The `probabilities` data you've already calculated will be useful and can be turned into a pandas DataFrame if desired (see next cell).
Good luck!  Share your results with the class!


```python
# create a DataFrame of log likelihoods for the test word items
df_probs = pd.DataFrame(data=probabilities)
df_probs.head()
```
