export type Difficulty = "Easy" | "Medium";

export type TestCase = {
  args: string[]; // one Python expression per parameter
  expected: string; // Python expression
};

export type Problem = {
  id: string;
  title: string;
  difficulty: Difficulty;
  description: string;
  fnName: string;
  params: string[];
  starter: string;
  cases: TestCase[];
  // Optional, for drawing recursive solutions. Expressions run inside each call of the
  // learner's code; the first that works is used, so list common variable names.
  traceLabel?: string[]; // what to show in each node, e.g. "sum(subset)"
  traceLabelText?: string; // legend text for the label
  traceGoal?: string; // may use `label`, e.g. "label == amount"; true turns the node green
  traceGoalText?: string; // legend text for the goal
};

// Index 0 is a warm-up. Indexes 1-30 are the main set.
export const PROBLEMS: Problem[] = [
  {
    "id": "second-largest",
    "title": "Second Largest",
    "difficulty": "Easy",
    "description": "Given a list of integers, return the second largest distinct value. The list always has at least two distinct values.",
    "fnName": "second_largest",
    "params": [
      "nums"
    ],
    "starter": "def second_largest(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 2, 3]"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[5, 5, 4]"
        ],
        "expected": "4"
      },
      {
        "args": [
          "[10, 9]"
        ],
        "expected": "9"
      },
      {
        "args": [
          "[-1, -5, -3]"
        ],
        "expected": "-3"
      },
      {
        "args": [
          "[4, 4, 4, 3]"
        ],
        "expected": "3"
      }
    ]
  },
  {
    "id": "two-sum",
    "title": "Two Sum",
    "difficulty": "Easy",
    "description": "Return the indices [i, j] with i < j of the two numbers in nums that add up to target. Exactly one such pair exists.",
    "fnName": "two_sum",
    "params": [
      "nums",
      "target"
    ],
    "starter": "def two_sum(nums, target):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[2, 7, 11, 15]",
          "9"
        ],
        "expected": "[0, 1]"
      },
      {
        "args": [
          "[3, 2, 4]",
          "6"
        ],
        "expected": "[1, 2]"
      },
      {
        "args": [
          "[3, 3]",
          "6"
        ],
        "expected": "[0, 1]"
      },
      {
        "args": [
          "[-1, -2, -3, -4, -5]",
          "-8"
        ],
        "expected": "[2, 4]"
      },
      {
        "args": [
          "[1, 5, 8, 3]",
          "11"
        ],
        "expected": "[2, 3]"
      }
    ]
  },
  {
    "id": "palindrome-number",
    "title": "Palindrome Number",
    "difficulty": "Easy",
    "description": "Return True if the integer x reads the same forwards and backwards. Negative numbers are never palindromes.",
    "fnName": "is_palindrome_number",
    "params": [
      "x"
    ],
    "starter": "def is_palindrome_number(x):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "121"
        ],
        "expected": "True"
      },
      {
        "args": [
          "-121"
        ],
        "expected": "False"
      },
      {
        "args": [
          "10"
        ],
        "expected": "False"
      },
      {
        "args": [
          "0"
        ],
        "expected": "True"
      },
      {
        "args": [
          "12321"
        ],
        "expected": "True"
      }
    ]
  },
  {
    "id": "roman-to-integer",
    "title": "Roman to Integer",
    "difficulty": "Easy",
    "description": "Convert a Roman numeral string to an integer. Symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000. A smaller symbol before a larger one is subtracted (IV = 4, IX = 9).",
    "fnName": "roman_to_int",
    "params": [
      "s"
    ],
    "starter": "def roman_to_int(s):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "'III'"
        ],
        "expected": "3"
      },
      {
        "args": [
          "'LVIII'"
        ],
        "expected": "58"
      },
      {
        "args": [
          "'MCMXCIV'"
        ],
        "expected": "1994"
      },
      {
        "args": [
          "'IV'"
        ],
        "expected": "4"
      },
      {
        "args": [
          "'IX'"
        ],
        "expected": "9"
      }
    ]
  },
  {
    "id": "valid-parentheses",
    "title": "Valid Parentheses",
    "difficulty": "Easy",
    "description": "Given a string containing only ()[]{} characters, return True if every bracket is closed by the same type in the correct order.",
    "fnName": "valid_parentheses",
    "params": [
      "s"
    ],
    "starter": "def valid_parentheses(s):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "'()'"
        ],
        "expected": "True"
      },
      {
        "args": [
          "'()[]{}'"
        ],
        "expected": "True"
      },
      {
        "args": [
          "'(]'"
        ],
        "expected": "False"
      },
      {
        "args": [
          "'([)]'"
        ],
        "expected": "False"
      },
      {
        "args": [
          "'{[]}'"
        ],
        "expected": "True"
      }
    ]
  },
  {
    "id": "longest-common-prefix",
    "title": "Longest Common Prefix",
    "difficulty": "Easy",
    "description": "Return the longest prefix shared by every string in the list, or an empty string if there is none.",
    "fnName": "longest_common_prefix",
    "params": [
      "strs"
    ],
    "starter": "def longest_common_prefix(strs):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "['flower', 'flow', 'flight']"
        ],
        "expected": "'fl'"
      },
      {
        "args": [
          "['dog', 'racecar', 'car']"
        ],
        "expected": "''"
      },
      {
        "args": [
          "['a']"
        ],
        "expected": "'a'"
      },
      {
        "args": [
          "['interview', 'internet', 'interval']"
        ],
        "expected": "'inter'"
      },
      {
        "args": [
          "['same', 'same']"
        ],
        "expected": "'same'"
      }
    ]
  },
  {
    "id": "remove-duplicates",
    "title": "Remove Duplicates",
    "difficulty": "Easy",
    "description": "Given a sorted list, return a new list with each value appearing only once, keeping the original order.",
    "fnName": "remove_duplicates",
    "params": [
      "nums"
    ],
    "starter": "def remove_duplicates(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 1, 2]"
        ],
        "expected": "[1, 2]"
      },
      {
        "args": [
          "[0, 0, 1, 1, 1, 2, 2, 3, 3, 4]"
        ],
        "expected": "[0, 1, 2, 3, 4]"
      },
      {
        "args": [
          "[5]"
        ],
        "expected": "[5]"
      },
      {
        "args": [
          "[2, 2, 2]"
        ],
        "expected": "[2]"
      },
      {
        "args": [
          "[1, 2, 3]"
        ],
        "expected": "[1, 2, 3]"
      }
    ]
  },
  {
    "id": "search-insert-position",
    "title": "Search Insert Position",
    "difficulty": "Easy",
    "description": "Given a sorted list of distinct integers, return the index of target. If it is missing, return the index where it would be inserted to keep the list sorted.",
    "fnName": "search_insert",
    "params": [
      "nums",
      "target"
    ],
    "starter": "def search_insert(nums, target):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 3, 5, 6]",
          "5"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[1, 3, 5, 6]",
          "2"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[1, 3, 5, 6]",
          "7"
        ],
        "expected": "4"
      },
      {
        "args": [
          "[1, 3, 5, 6]",
          "0"
        ],
        "expected": "0"
      },
      {
        "args": [
          "[1]",
          "1"
        ],
        "expected": "0"
      }
    ]
  },
  {
    "id": "maximum-subarray",
    "title": "Maximum Subarray",
    "difficulty": "Easy",
    "description": "Return the largest sum of any non-empty contiguous subarray of nums.",
    "fnName": "max_subarray",
    "params": [
      "nums"
    ],
    "starter": "def max_subarray(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[-2, 1, -3, 4, -1, 2, 1, -5, 4]"
        ],
        "expected": "6"
      },
      {
        "args": [
          "[1]"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[5, 4, -1, 7, 8]"
        ],
        "expected": "23"
      },
      {
        "args": [
          "[-3, -1, -2]"
        ],
        "expected": "-1"
      },
      {
        "args": [
          "[0, 0, 0]"
        ],
        "expected": "0"
      }
    ]
  },
  {
    "id": "plus-one",
    "title": "Plus One",
    "difficulty": "Easy",
    "description": "digits is a non-negative integer written as a list of decimal digits, most significant first. Return the digits of that integer plus one.",
    "fnName": "plus_one",
    "params": [
      "digits"
    ],
    "starter": "def plus_one(digits):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 2, 3]"
        ],
        "expected": "[1, 2, 4]"
      },
      {
        "args": [
          "[4, 3, 2, 1]"
        ],
        "expected": "[4, 3, 2, 2]"
      },
      {
        "args": [
          "[9]"
        ],
        "expected": "[1, 0]"
      },
      {
        "args": [
          "[9, 9, 9]"
        ],
        "expected": "[1, 0, 0, 0]"
      },
      {
        "args": [
          "[0]"
        ],
        "expected": "[1]"
      }
    ]
  },
  {
    "id": "climbing-stairs",
    "title": "Climbing Stairs",
    "difficulty": "Easy",
    "description": "You climb a staircase of n steps, taking 1 or 2 steps at a time. Return the number of distinct ways to reach the top.",
    "fnName": "climb_stairs",
    "params": [
      "n"
    ],
    "starter": "def climb_stairs(n):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "1"
        ],
        "expected": "1"
      },
      {
        "args": [
          "2"
        ],
        "expected": "2"
      },
      {
        "args": [
          "3"
        ],
        "expected": "3"
      },
      {
        "args": [
          "5"
        ],
        "expected": "8"
      },
      {
        "args": [
          "10"
        ],
        "expected": "89"
      }
    ]
  },
  {
    "id": "merge-sorted-lists",
    "title": "Merge Sorted Lists",
    "difficulty": "Easy",
    "description": "Given two sorted lists, return a single sorted list containing all of their elements.",
    "fnName": "merge_sorted",
    "params": [
      "a",
      "b"
    ],
    "starter": "def merge_sorted(a, b):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 2, 3]",
          "[2, 5, 6]"
        ],
        "expected": "[1, 2, 2, 3, 5, 6]"
      },
      {
        "args": [
          "[1]",
          "[]"
        ],
        "expected": "[1]"
      },
      {
        "args": [
          "[]",
          "[0]"
        ],
        "expected": "[0]"
      },
      {
        "args": [
          "[1, 3, 5]",
          "[2, 4, 6]"
        ],
        "expected": "[1, 2, 3, 4, 5, 6]"
      },
      {
        "args": [
          "[-1, 0]",
          "[-3, -2]"
        ],
        "expected": "[-3, -2, -1, 0]"
      }
    ]
  },
  {
    "id": "best-time-to-buy-and-sell-stock",
    "title": "Best Time to Buy and Sell Stock",
    "difficulty": "Easy",
    "description": "prices[i] is a stock price on day i. Buy on one day and sell on a later day. Return the maximum profit, or 0 if no profit is possible.",
    "fnName": "max_profit",
    "params": [
      "prices"
    ],
    "starter": "def max_profit(prices):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[7, 1, 5, 3, 6, 4]"
        ],
        "expected": "5"
      },
      {
        "args": [
          "[7, 6, 4, 3, 1]"
        ],
        "expected": "0"
      },
      {
        "args": [
          "[1, 2]"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[2, 4, 1]"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[3, 3, 3]"
        ],
        "expected": "0"
      }
    ]
  },
  {
    "id": "valid-palindrome",
    "title": "Valid Palindrome",
    "difficulty": "Easy",
    "description": "Return True if s is a palindrome after lowercasing it and removing every character that is not a letter or digit.",
    "fnName": "is_palindrome",
    "params": [
      "s"
    ],
    "starter": "def is_palindrome(s):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "'A man, a plan, a canal: Panama'"
        ],
        "expected": "True"
      },
      {
        "args": [
          "'race a car'"
        ],
        "expected": "False"
      },
      {
        "args": [
          "' '"
        ],
        "expected": "True"
      },
      {
        "args": [
          "'0P'"
        ],
        "expected": "False"
      },
      {
        "args": [
          "'No lemon, no melon'"
        ],
        "expected": "True"
      }
    ]
  },
  {
    "id": "single-number",
    "title": "Single Number",
    "difficulty": "Easy",
    "description": "Every value in nums appears exactly twice except one. Return the value that appears once.",
    "fnName": "single_number",
    "params": [
      "nums"
    ],
    "starter": "def single_number(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[2, 2, 1]"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[4, 1, 2, 1, 2]"
        ],
        "expected": "4"
      },
      {
        "args": [
          "[1]"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[-1, -1, -2]"
        ],
        "expected": "-2"
      },
      {
        "args": [
          "[7, 3, 7]"
        ],
        "expected": "3"
      }
    ]
  },
  {
    "id": "contains-duplicate",
    "title": "Contains Duplicate",
    "difficulty": "Easy",
    "description": "Return True if any value appears at least twice in nums, otherwise False.",
    "fnName": "contains_duplicate",
    "params": [
      "nums"
    ],
    "starter": "def contains_duplicate(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 2, 3, 1]"
        ],
        "expected": "True"
      },
      {
        "args": [
          "[1, 2, 3, 4]"
        ],
        "expected": "False"
      },
      {
        "args": [
          "[1, 1, 1, 3, 3, 4, 3, 2, 4, 2]"
        ],
        "expected": "True"
      },
      {
        "args": [
          "[0]"
        ],
        "expected": "False"
      },
      {
        "args": [
          "[]"
        ],
        "expected": "False"
      }
    ]
  },
  {
    "id": "valid-anagram",
    "title": "Valid Anagram",
    "difficulty": "Easy",
    "description": "Return True if t is an anagram of s, meaning it uses exactly the same letters with the same counts.",
    "fnName": "is_anagram",
    "params": [
      "s",
      "t"
    ],
    "starter": "def is_anagram(s, t):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "'anagram'",
          "'nagaram'"
        ],
        "expected": "True"
      },
      {
        "args": [
          "'rat'",
          "'car'"
        ],
        "expected": "False"
      },
      {
        "args": [
          "'a'",
          "'a'"
        ],
        "expected": "True"
      },
      {
        "args": [
          "'ab'",
          "'a'"
        ],
        "expected": "False"
      },
      {
        "args": [
          "'listen'",
          "'silent'"
        ],
        "expected": "True"
      }
    ]
  },
  {
    "id": "missing-number",
    "title": "Missing Number",
    "difficulty": "Easy",
    "description": "nums holds n distinct integers taken from the range 0..n. Return the one number in that range that is missing.",
    "fnName": "missing_number",
    "params": [
      "nums"
    ],
    "starter": "def missing_number(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[3, 0, 1]"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[0, 1]"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[9, 6, 4, 2, 3, 5, 7, 0, 1]"
        ],
        "expected": "8"
      },
      {
        "args": [
          "[0]"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[1]"
        ],
        "expected": "0"
      }
    ]
  },
  {
    "id": "move-zeroes",
    "title": "Move Zeroes",
    "difficulty": "Easy",
    "description": "Return a new list with every 0 moved to the end while the other values keep their relative order.",
    "fnName": "move_zeroes",
    "params": [
      "nums"
    ],
    "starter": "def move_zeroes(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[0, 1, 0, 3, 12]"
        ],
        "expected": "[1, 3, 12, 0, 0]"
      },
      {
        "args": [
          "[0]"
        ],
        "expected": "[0]"
      },
      {
        "args": [
          "[1, 2, 3]"
        ],
        "expected": "[1, 2, 3]"
      },
      {
        "args": [
          "[0, 0, 1]"
        ],
        "expected": "[1, 0, 0]"
      },
      {
        "args": [
          "[4, 0, 5, 0, 0, 6]"
        ],
        "expected": "[4, 5, 6, 0, 0, 0]"
      }
    ]
  },
  {
    "id": "fizz-buzz",
    "title": "Fizz Buzz",
    "difficulty": "Easy",
    "description": "Return a list of strings for the numbers 1 to n. Use \"FizzBuzz\" for multiples of both 3 and 5, \"Fizz\" for multiples of 3, \"Buzz\" for multiples of 5, and the number as a string otherwise.",
    "fnName": "fizz_buzz",
    "params": [
      "n"
    ],
    "starter": "def fizz_buzz(n):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "1"
        ],
        "expected": "['1']"
      },
      {
        "args": [
          "3"
        ],
        "expected": "['1', '2', 'Fizz']"
      },
      {
        "args": [
          "5"
        ],
        "expected": "['1', '2', 'Fizz', '4', 'Buzz']"
      },
      {
        "args": [
          "6"
        ],
        "expected": "['1', '2', 'Fizz', '4', 'Buzz', 'Fizz']"
      },
      {
        "args": [
          "10"
        ],
        "expected": "['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', 'Fizz', 'Buzz']"
      },
      {
        "args": [
          "15"
        ],
        "expected": "['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', 'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']"
      }
    ]
  },
  {
    "id": "product-of-array-except-self",
    "title": "Product of Array Except Self",
    "difficulty": "Medium",
    "description": "Return a list where element i is the product of every value in nums except nums[i]. Do not use division.",
    "fnName": "product_except_self",
    "params": [
      "nums"
    ],
    "starter": "def product_except_self(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 2, 3, 4]"
        ],
        "expected": "[24, 12, 8, 6]"
      },
      {
        "args": [
          "[-1, 1, 0, -3, 3]"
        ],
        "expected": "[0, 0, 9, 0, 0]"
      },
      {
        "args": [
          "[2, 3]"
        ],
        "expected": "[3, 2]"
      },
      {
        "args": [
          "[1, 1, 1]"
        ],
        "expected": "[1, 1, 1]"
      },
      {
        "args": [
          "[5, 0, 2]"
        ],
        "expected": "[0, 10, 0]"
      }
    ]
  },
  {
    "id": "3sum",
    "title": "3Sum",
    "difficulty": "Medium",
    "description": "Return every unique triplet of values that sums to 0. Sort each triplet in ascending order, and sort the list of triplets in ascending order. No triplet may appear twice.",
    "fnName": "three_sum",
    "params": [
      "nums"
    ],
    "starter": "def three_sum(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[-1, 0, 1, 2, -1, -4]"
        ],
        "expected": "[[-1, -1, 2], [-1, 0, 1]]"
      },
      {
        "args": [
          "[0, 1, 1]"
        ],
        "expected": "[]"
      },
      {
        "args": [
          "[0, 0, 0]"
        ],
        "expected": "[[0, 0, 0]]"
      },
      {
        "args": [
          "[-2, 0, 1, 1, 2]"
        ],
        "expected": "[[-2, 0, 2], [-2, 1, 1]]"
      },
      {
        "args": [
          "[1, 2, -2, -1]"
        ],
        "expected": "[]"
      }
    ]
  },
  {
    "id": "longest-substring-without-repeating-characters",
    "title": "Longest Substring Without Repeating Characters",
    "difficulty": "Medium",
    "description": "Return the length of the longest substring of s that contains no repeated characters.",
    "fnName": "length_of_longest_substring",
    "params": [
      "s"
    ],
    "starter": "def length_of_longest_substring(s):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "'abcabcbb'"
        ],
        "expected": "3"
      },
      {
        "args": [
          "'bbbbb'"
        ],
        "expected": "1"
      },
      {
        "args": [
          "'pwwkew'"
        ],
        "expected": "3"
      },
      {
        "args": [
          "''"
        ],
        "expected": "0"
      },
      {
        "args": [
          "'dvdf'"
        ],
        "expected": "3"
      }
    ]
  },
  {
    "id": "container-with-most-water",
    "title": "Container With Most Water",
    "difficulty": "Medium",
    "description": "height[i] is the height of a vertical line at position i. Pick two lines that, together with the x-axis, hold the most water. Return that area (distance between the lines times the shorter height).",
    "fnName": "max_area",
    "params": [
      "height"
    ],
    "starter": "def max_area(height):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 8, 6, 2, 5, 4, 8, 3, 7]"
        ],
        "expected": "49"
      },
      {
        "args": [
          "[1, 1]"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[4, 3, 2, 1, 4]"
        ],
        "expected": "16"
      },
      {
        "args": [
          "[1, 2, 1]"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[2, 3, 4, 5, 18, 17, 6]"
        ],
        "expected": "17"
      }
    ]
  },
  {
    "id": "rotate-array",
    "title": "Rotate Array",
    "difficulty": "Medium",
    "description": "Return a new list that is nums rotated to the right by k positions. k may be larger than the length of the list.",
    "fnName": "rotate",
    "params": [
      "nums",
      "k"
    ],
    "starter": "def rotate(nums, k):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 2, 3, 4, 5, 6, 7]",
          "3"
        ],
        "expected": "[5, 6, 7, 1, 2, 3, 4]"
      },
      {
        "args": [
          "[-1, -100, 3, 99]",
          "2"
        ],
        "expected": "[3, 99, -1, -100]"
      },
      {
        "args": [
          "[1]",
          "5"
        ],
        "expected": "[1]"
      },
      {
        "args": [
          "[1, 2]",
          "3"
        ],
        "expected": "[2, 1]"
      },
      {
        "args": [
          "[1, 2, 3]",
          "0"
        ],
        "expected": "[1, 2, 3]"
      }
    ]
  },
  {
    "id": "coin-change",
    "traceLabel": [
      "sum(subset)", "sum(path)", "sum(curr)", "sum(cur)", "sum(current)", "sum(combo)",
      "sum(comb)", "sum(chosen)", "sum(stack)", "sum(coins_used)",
      "curr_sum", "cur_sum", "current_sum", "total", "amount - remaining", "amount - rem"
    ],
    "traceLabelText": "sum of the coins chosen so far",
    "traceGoal": "label == amount",
    "traceGoalText": "sum == amount",
    "title": "Coin Change",
    "difficulty": "Medium",
    "description": "Given coin denominations and a target amount, return the fewest coins needed to make that amount, or -1 if it is impossible. You have unlimited coins of each kind.",
    "fnName": "coin_change",
    "params": [
      "coins",
      "amount"
    ],
    "starter": "def coin_change(coins, amount):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 2, 5]",
          "11"
        ],
        "expected": "3"
      },
      {
        "args": [
          "[2]",
          "3"
        ],
        "expected": "-1"
      },
      {
        "args": [
          "[1]",
          "0"
        ],
        "expected": "0"
      },
      {
        "args": [
          "[1, 3, 4]",
          "6"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[5, 10]",
          "15"
        ],
        "expected": "2"
      }
    ]
  },
  {
    "id": "house-robber",
    "title": "House Robber",
    "difficulty": "Medium",
    "description": "nums[i] is the money in house i along a street. You cannot rob two adjacent houses. Return the maximum total you can rob.",
    "fnName": "rob",
    "params": [
      "nums"
    ],
    "starter": "def rob(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 2, 3, 1]"
        ],
        "expected": "4"
      },
      {
        "args": [
          "[2, 7, 9, 3, 1]"
        ],
        "expected": "12"
      },
      {
        "args": [
          "[5]"
        ],
        "expected": "5"
      },
      {
        "args": [
          "[2, 1, 1, 2]"
        ],
        "expected": "4"
      },
      {
        "args": [
          "[0, 0]"
        ],
        "expected": "0"
      }
    ]
  },
  {
    "id": "longest-increasing-subsequence",
    "title": "Longest Increasing Subsequence",
    "difficulty": "Medium",
    "description": "Return the length of the longest strictly increasing subsequence of nums. A subsequence keeps the original order but may skip elements.",
    "fnName": "length_of_lis",
    "params": [
      "nums"
    ],
    "starter": "def length_of_lis(nums):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[10, 9, 2, 5, 3, 7, 101, 18]"
        ],
        "expected": "4"
      },
      {
        "args": [
          "[0, 1, 0, 3, 2, 3]"
        ],
        "expected": "4"
      },
      {
        "args": [
          "[7, 7, 7, 7]"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[1, 2, 3, 4, 5]"
        ],
        "expected": "5"
      },
      {
        "args": [
          "[5, 4, 3]"
        ],
        "expected": "1"
      }
    ]
  },
  {
    "id": "search-in-rotated-sorted-array",
    "title": "Search in Rotated Sorted Array",
    "difficulty": "Medium",
    "description": "nums was a sorted list of distinct values that has been rotated at an unknown pivot (for example [4,5,6,7,0,1,2]). Return the index of target, or -1 if it is not present.",
    "fnName": "search",
    "params": [
      "nums",
      "target"
    ],
    "starter": "def search(nums, target):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[4, 5, 6, 7, 0, 1, 2]",
          "0"
        ],
        "expected": "4"
      },
      {
        "args": [
          "[4, 5, 6, 7, 0, 1, 2]",
          "3"
        ],
        "expected": "-1"
      },
      {
        "args": [
          "[1]",
          "0"
        ],
        "expected": "-1"
      },
      {
        "args": [
          "[3, 1]",
          "1"
        ],
        "expected": "1"
      },
      {
        "args": [
          "[5, 1, 3]",
          "5"
        ],
        "expected": "0"
      }
    ]
  },
  {
    "id": "merge-intervals",
    "title": "Merge Intervals",
    "difficulty": "Medium",
    "description": "Merge all overlapping intervals (touching endpoints count as overlapping) and return the result sorted by start.",
    "fnName": "merge_intervals",
    "params": [
      "intervals"
    ],
    "starter": "def merge_intervals(intervals):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[[1, 3], [2, 6], [8, 10], [15, 18]]"
        ],
        "expected": "[[1, 6], [8, 10], [15, 18]]"
      },
      {
        "args": [
          "[[1, 4], [4, 5]]"
        ],
        "expected": "[[1, 5]]"
      },
      {
        "args": [
          "[[1, 4], [0, 4]]"
        ],
        "expected": "[[0, 4]]"
      },
      {
        "args": [
          "[[1, 4], [2, 3]]"
        ],
        "expected": "[[1, 4]]"
      },
      {
        "args": [
          "[[5, 6], [1, 2], [3, 4]]"
        ],
        "expected": "[[1, 2], [3, 4], [5, 6]]"
      }
    ]
  },
  {
    "id": "subarray-sum-equals-k",
    "title": "Subarray Sum Equals K",
    "difficulty": "Medium",
    "description": "Return how many contiguous, non-empty subarrays of nums add up to k.",
    "fnName": "subarray_sum",
    "params": [
      "nums",
      "k"
    ],
    "starter": "def subarray_sum(nums, k):\n    # your code here\n    pass\n",
    "cases": [
      {
        "args": [
          "[1, 1, 1]",
          "2"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[1, 2, 3]",
          "3"
        ],
        "expected": "2"
      },
      {
        "args": [
          "[1, -1, 0]",
          "0"
        ],
        "expected": "3"
      },
      {
        "args": [
          "[1]",
          "0"
        ],
        "expected": "0"
      },
      {
        "args": [
          "[3, 4, 7, 2, -3, 1, 4, 2]",
          "7"
        ],
        "expected": "4"
      }
    ]
  }
];