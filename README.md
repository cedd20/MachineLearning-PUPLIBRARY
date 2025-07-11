# MachineLearning-PUPLIBRARY

## Abstract

This project implements a modern, user-friendly Library Book Recommendation and Analytics System for the Polytechnic University of the Philippines (PUP) Library, leveraging data mining and interactive visualization techniques.

- Tech Stack: Python, Tkinter, pandas, matplotlib, mlxtend  
- Focus: Market basket analysis for personalized recommendations  
- Highlights: Clean UI/UX, interactive visualizations, course-aware suggestions

---

## Core Logic & Algorithms

### Data Preprocessing

- Loads book borrowing transaction data
- Groups books by transaction, labeled with course and year
- Extracts unique book titles and course/year combinations for autocomplete fields

### Book Recommendation (Apriori Algorithm)

- Uses Apriori algorithm (via `mlxtend`) to find frequent itemsets and association rules
- Recommendation flow:
  - Filter transactions by selected course/year
  - Apply Apriori to find relevant association rules
  - Recommend top 3 books borrowed with the queried one, based on:
    - Confidence
    - Lift
- Fallback: If no strong rules are found, recommends the most borrowed books for that course/year

### Book Category Analytics (Program-Based)

- Groups books by the program that borrows them most frequently (e.g., BSIT, BSA)
- Assigns each book to its dominant program based on borrowing frequency
- Maps programs to descriptive names (e.g., "Books about Information Science")
- Visualizations include:
  - Pie chart with a maroon-yellow color palette
  - Grouped table listing books per category

### Interactive Visualization

- Pie chart includes hoverable tooltips (via `mplcursors`) showing:
  - Category name
  - Book count
  - Borrowing percentage
- Layout: Combined chart and table in one UI window for a modern experience

### Modern UI/UX with Tkinter

- Clean, adaptive interface with:
  - Autocomplete input fields
  - Styled buttons
  - Organized main window with access to all features (search, recommend, analytics)

---

## Use Cases

### For Students

- Search for books and receive personalized recommendations
- Discover popular titles among peers from the same course and year

### For Librarians / Administrators

- Analyze borrowing patterns by academic program
- Identify course-specific book trends
- Visualize collection diversity to inform decisions

### For Data-Driven Strategy

- Use Apriori to uncover borrowing behavior patterns
- Present actionable insights through interactive charts and tables

---

## Summary

This system blends classic market basket analysis with a modern, intuitive user experience. It enables:

- Personalized book suggestions for students
- Data-driven insights for library management
- A comprehensive, accessible platform for educational resource discovery
