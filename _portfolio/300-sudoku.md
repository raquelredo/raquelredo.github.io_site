---
title: "Solving Sudokus"
layout: "post"
excerpt: "First project for the Artificial Intelligence Nanodegree. Coding how to Solve a Sudoku puzzle game."
tags: [python, Artificial Intelligence, sudoku, naked twins, diagonal Sudoku, Udacity]
header:
  teaser: sudoku_colox.jpg
link:
share: true
categories: portfolio
---

# Solving a Sudoku

# Introduction and purpose
The following project has been done while studying the Artificial Intelligence Nanodegre at [Udacity][bc887b54]. This is the first project I needed to submit. As because of that, most of the defining functions and solving methodology are not complex but have been selected in order to understand the basic foundations and algorithms.

  [bc887b54]: www.Udacity.com "Udacity"

The puzzle itself is nothing more than a grid of little boxes. They are stacked nine high and nine wide, making 81 cells total. The puzzle comes with some of the cells (usually less than half of them) already filled in. The object of the game is simple: fill in the empty cells!

Easy, right? Well, hang on a sec...there is one rule that we must follow: no repeats are allowed in any row, column, or block. To put it another way - all nine numbers must be used in each row, column, and block.

For this project specifically, we need to use as well a more advanced rule. Numbers should not be repeated as well within the diagonal containing that box.

As for me, it was my father who showed how to fill a Sudoku game using a pen, algorithmic thinking as well as attention to detail. Now I have the chance to leave the pen behind, code the solving, deepen my Python coding skills as well as introducing myself much more into programmatic thinking.

# Requirements
The requirements for this project have been collected on the [requirements](https://github.com/raquelredo/raquelredo.github.io_site/blob/master/_portfolio/AI-Sudoku/requirements.txt) file.

# Naming Our Objects
The first step is to set up the board game and define the objects we are going to need.

That will be our board game:

![](https://github.com/raquelredo/raquelredo.github.io_site/blob/master/_portfolio/AI-Sudoku/sudoku-board-bare.jpg?raw=true)

```python
import itertools # for most efficient loops
```
We will set the assignments array:

```python
assignments = []
```
and will create the arrows and name mapping:

```pythonx
rows = 'ABCDEFGHI'
cols = '123456789'
```
A box will be a cell named with a letter from the corresponding row from `rows` and a number from the corresponding column `cols`.
```python
def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]
```
```python
boxes = cross(rows,cols)
```
```python
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
```
For this Sudouku we have been asked to consider a variant called **Diagonal Sudokou**, which consists in adding diagonals as units in which numbers should not be repeated.

```python
## Diagonal units
diagonal_units =  [[s+t for s,t in zip(rows,cols)],[s+t for s,t in zip(rows,cols[::-1])]]
```
and we will set the objects as follows:
```python
unitlist = row_units + column_units + square_units + diagonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)
```
Each row, column, diagonal and a 3x3 square will be an `unit`.
For each box there is a `peer` which is another box from the same unit.

![](https://github.com/raquelredo/raquelredo.github.io_site/blob/master/_portfolio/AI-Sudoku/diagonal-sudoku.png?raw=true)

```python
def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values
  ```
## The twins strategy!
​Me, as a happily born twin myself, discovered soon that working in pairs it is a good way to solve problems as well as make it more fun.
The `naked twins` strategy consists in tracking pairs of identical values in two boxes, and remove that values for the other peers as the values can only be set in that two boxes. By doing this, we will reduce our set of values, and set it ready for next step.

```python
def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}
    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    # Steps:
    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers
    for unit in unitlist:
        # Find all boxes with two digits as data set
        pairs = [box for box in unit if len(values[box]) == 2]
        # make all possible combinations
        possible_twins = [list(pair) for pair in itertools.combinations(pairs, 2)]
        for pair in possible_twins:
            box1 = pair[0]
            box2 = pair[1]
            # Find the naked twins
            if values[box1] == values[box2]:
                for box in unit:
                    # Eliminate the naked twins from their peers
                    if box != box1 and box != box2:
                        for digit in values[box1]:
                            values[box] = values[box].replace(digit,'')
    return values
```
```python
def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    chars = []
    digits = '123456789'
    for c in grid:
        if c in digits:
            chars.append(c)
        if c == '.':
            chars.append(digits)
    assert len(chars) == 81
    return dict(zip(boxes, chars))
```
This chunk of code is for displaying the sudoku:
```python
def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return
```
After identifying the naked twins, we should eliminate the values from its peers.
```python
def eliminate(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values

def only_choice(values):
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values
```
Now we must set the rules for iterating the strategies finding the solution.
```python
def reduce_puzzle(values):
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values
```
```python
def search(values):
    values = reduce_puzzle(values)
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in boxes):
        return values #That means is solved!!
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus
    for value in values[s]:
        sudoku2 = values.copy()
        sudoku2[s] = value
        attempt = search(sudoku2)
        if attempt:
            return attempt
```
```python
def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """

    values = grid_values(grid)
    values = search(values)
    return values
```
Let's try it!
```python
if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
```
and this is the resulting Sudoku: ![](https://github.com/raquelredo/raquelredo.github.io_site/blob/master/_portfolio/AI-Sudoku/solution.JPG?raw=true)

Hooray!

# Conclusion
One of the main challenges I had written this code besides the fact that I needed to learn some new concepts as well as upgrade my python skills, is the fact that I reached a solution in my coding but was soooo inefficient that my machine runs out of memory trying to solve the puzzle. So I needed to recode all several times.That makes me realise that in AI that is one of the main points to be successful.
