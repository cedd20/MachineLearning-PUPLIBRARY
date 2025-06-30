import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import tkinter as tk
from tkinter import ttk, messagebox
import re
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import mplcursors
import matplotlib.gridspec as gridspec

# Load and preprocess data
DATA_PATH = 'C:/ML_Final_Project/PUP-Library-ML/resources/PUPLibDataset.csv'
df = pd.read_csv(DATA_PATH)

# Group by Transaction ID to get sets of books borrowed together
transactions = df.groupby('Transaction ID')['Book Title'].apply(list).tolist()

# Create one-hot encoded DataFrame for Apriori
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_ary, columns=te.columns_)

# Run Apriori
frequent_itemsets = apriori(df_te, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Add course and section info to each transaction for filtering
transaction_info = df.groupby('Transaction ID').agg({
    'Book Title': list,
    'Course and Year': 'first'
}).reset_index()

# Extract all unique book names
book_titles = sorted(df['Book Title'].unique().tolist())
# Extract all unique course and year values (e.g., BSIT-1, BSIT-2, ...)
course_year_list = sorted(df['Course and Year'].unique().tolist())

class AutocompleteEntry(tk.Entry):
    def __init__(self, book_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.book_list = book_list
        self.var = self["textvariable"] = tk.StringVar()
        self.var.trace('w', self.changed)
        self.bind("<Down>", self.move_down)
        self.bind("<Up>", self.move_up)
        self.bind("<Return>", self.selection)
        self.bind("<FocusOut>", lambda e: self.hide_listbox())
        self.lb = None
        self.lb_index = 0

    def changed(self, *args):
        value = self.var.get()
        if value == '':
            self.hide_listbox()
            return
        matches = [b for b in self.book_list if value.lower() in b.lower()]
        if not matches:
            matches = ["book not available"]
        self.show_listbox(matches)

    def show_listbox(self, matches):
        if self.lb:
            self.lb.destroy()
        self.lb = tk.Listbox(self.master, width=self["width"])
        for m in matches:
            self.lb.insert(tk.END, m)
        self.lb.place(x=self.winfo_x(), y=self.winfo_y()+self.winfo_height())
        self.lb.bind("<Button-1>", self.selection)
        self.lb_index = 0
        self.lb.select_set(self.lb_index)

    def hide_listbox(self):
        if self.lb:
            self.lb.destroy()
            self.lb = None

    def move_down(self, event):
        if self.lb and self.lb_index < self.lb.size() - 1:
            self.lb_index += 1
            self.lb.select_clear(0, tk.END)
            self.lb.select_set(self.lb_index)
        return "break"

    def move_up(self, event):
        if self.lb and self.lb_index > 0:
            self.lb_index -= 1
            self.lb.select_clear(0, tk.END)
            self.lb.select_set(self.lb_index)
        return "break"

    def selection(self, event=None):
        if self.lb:
            value = self.lb.get(self.lb_index)
            if value != "book not available":
                self.var.set(value)
            self.hide_listbox()
        return "break"

# Tkinter UI
class LibraryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PUP Library Book Recommendation App")
        self.root.geometry('800x600')

        # Input fields
        tk.Label(root, text="Book Name:").pack(pady=5)
        self.book_entry = AutocompleteEntry(book_titles, root, width=50)
        self.book_entry.pack(pady=5)

        tk.Label(root, text="Course and Year (e.g., BSIT-1):").pack(pady=5)
        self.course_entry = AutocompleteEntry(course_year_list, root, width=30)
        self.course_entry.pack(pady=5)

        tk.Button(root, text="Search & Recommend", command=self.recommend).pack(pady=10)
        tk.Button(root, text="Show Book Categories Visualization", command=self.show_categories_viz).pack(pady=10)

        # Results area
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(self.result_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def recommend(self):
        book = self.book_entry.get().strip()
        course_year = self.course_entry.get().strip()
        if not book or book == "book not available":
            messagebox.showerror("Input Error", "Please enter a valid book name.")
            return
        if not course_year or course_year == "book not available":
            messagebox.showerror("Input Error", "Please enter a valid course and year.")
            return

        # Filter transactions by exact course and year
        filtered_tids = transaction_info[transaction_info['Course and Year'].str.upper() == course_year.upper()]['Book Title']
        filtered_transactions = filtered_tids.tolist()
        rules_f = None
        if filtered_transactions:
            te_ary_f = te.transform(filtered_transactions)
            df_te_f = pd.DataFrame(te_ary_f, columns=list(te.columns_))  # type: ignore
            frequent_itemsets_f = apriori(df_te_f, min_support=0.01, use_colnames=True)
            rules_f = association_rules(frequent_itemsets_f, metric="confidence", min_threshold=0.1)
        if rules_f is not None and not rules_f.empty:
            rec_rules = rules_f[rules_f['antecedents'].apply(lambda x: book in x)]
        else:
            rec_rules = pd.DataFrame()

        if rec_rules is not None and not rec_rules.empty:
            # Sort by confidence and lift, get top 3
            rec_rules = rec_rules.sort_values(['confidence', 'lift'], ascending=[False, False]).head(3)  # type: ignore
            result = f"Top 3 recommendations for '{book}' (based on BS Apriori):\n\n"
            for idx, row in rec_rules.iterrows():
                consequents = ', '.join(row['consequents'])
                result += f"Recommended Book: {consequents}\n"
                result += f"  - Confidence: {row['confidence']:.2f}\n"
                result += f"  - Support: {row['support']:.2f}\n"
                result += f"  - Lift: {row['lift']:.2f}\n"
                result += f"  - Analytics: If a student borrows '{book}', they are likely to also borrow '{consequents}' (confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f}).\n\n"
            self.display_result(result)
            return
        # If no Apriori recommendations, recommend top 3 most borrowed books by classmates (excluding the searched book)
        classmates_books = df[df['Course and Year'].str.upper() == course_year.upper()]
        book_series = classmates_books['Book Title'].copy()
        top_books = book_series[book_series.str.lower() != book.lower()].value_counts().head(3)  # type: ignore
        if not top_books.empty:
            result = f"No strong association rules for '{book}'.\nTop 3 books most borrowed by students in {course_year}:\n\n"
            for b, count in top_books.items():
                result += f"Recommended Book: {b}\n  - Times Borrowed: {count}\n\n"
            self.display_result(result)
            return
        # If still nothing, show no recommendations
        self.display_result(f"No recommendations found for '{book}' in {course_year}.")

    def show_categories_viz(self):
        # Extract program prefix for each transaction
        df['Program'] = df['Course and Year'].str.extract(r'(^[A-Z]+)')
        program_list = df['Program'].unique()
        book_to_program = {}
        program_book_support = defaultdict(dict)
        # For each program, run Apriori on their transactions
        for prog in program_list:
            prog_df = df[df['Program'] == prog]
            prog_transactions = prog_df.groupby('Transaction ID')['Book Title'].apply(list).tolist()
            if not prog_transactions:
                continue
            te_prog = TransactionEncoder()
            te_ary_prog = te_prog.fit(prog_transactions).transform(prog_transactions)
            df_te_prog = pd.DataFrame(te_ary_prog, columns=list(te_prog.columns_))  # type: ignore
            freq_itemsets_prog = apriori(df_te_prog, min_support=0.01, use_colnames=True)
            # For each book, get its support in this program
            for _, row in freq_itemsets_prog.iterrows():
                items = row['itemsets']
                if len(items) == 1:
                    book = list(items)[0]
                    program_book_support[prog][book] = row['support']
        # Assign each book to the program where it has the highest support
        for book in book_titles:
            max_prog = None
            max_support = 0
            for prog in program_book_support:
                support = program_book_support[prog].get(book, 0)
                if support > max_support:
                    max_support = support
                    max_prog = prog
            if max_prog:
                book_to_program[book] = max_prog
            else:
                book_to_program[book] = 'Other'
        # Count books per program
        mapping = {
            'BSIT': 'Information Science',
            'BSA': 'Accountancy',
            'BSOA': 'Office Administration',
            'BSHM': 'Hospitality Management',
            'BSCE': 'Civil Engineering',
            'BSEE': 'Electrical Engineering',
            'BSARCHI': 'Architecture',
            'BSAM': 'Applied Mathematics',
            'BSBIO': 'Biology',
            'BSND': 'Nutrition and Dietetics',
            'BEED': 'Elementary Education',
            'BSED': 'Secondary Education',
            'BPA': 'Public Administration',
            'BSBA': 'Business Administration',
            'DCpET': 'Information Science',
            'DEET': 'Electrical Engineering',
            'DCvET': 'Civil Engineering',
            'DICT': 'Information Science',
            'DOMT': 'Office Administration',
        }
        program_names = {p: f"Books about {self.program_full_name(p)}" for p in program_list}
        program_names['Other'] = 'Other'
        # Remove any unmapped or short prefixes (like 'DC') from program_names
        for k in list(program_names.keys()):
            if program_names[k] == f"Books about {k}" and k not in mapping:
                program_names[k] = 'Other'
        category_counts = Counter([program_names.get(book_to_program[b], 'Other') for b in book_titles])
        # Pie chart: top 6 categories, rest as 'Other'
        categories, counts = zip(*category_counts.most_common(6))
        other_count = sum(list(category_counts.values())[6:])
        if other_count > 0:
            categories = list(categories) + ['Other']
            counts = list(counts) + [other_count]
        total_books = sum(counts)
        # Color palette (maroon + yellow)
        pie_colors = ['#4D1414', '#7F0404', '#C46B02', '#F4BB00', '#FDDE54', '#B8860B', '#FFD700']
        # Group books by category for the table
        category_books = defaultdict(list)
        for book, prog in book_to_program.items():
            category_books[program_names.get(prog, 'Other')].append(book)
        # Prepare grouped table data: only show category name for first book in each group
        table_data = []
        for cat in categories:
            books_list = sorted(category_books[cat])
            for i, book in enumerate(books_list):
                if i == 0:
                    table_data.append([cat, book])
                else:
                    table_data.append(['', book])
        # Create figure with GridSpec
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5])
        # Pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        pie_result = ax1.pie(counts, labels=None, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 18}, colors=pie_colors[:len(counts)])
        if len(pie_result) == 3:
            wedges, texts, autotexts = pie_result
        else:
            wedges, texts = pie_result
        ax1.set_title('BOOK DISTRIBUTION', fontsize=24, fontweight='bold', pad=30)
        # Add interactive tooltips
        cursor = mplcursors.cursor(wedges, hover=True)
        def fmt_pie_slice(index):
            cat = categories[index]
            count = counts[index]
            percent = 100.0 * count / total_books
            return f"{cat}\nBooks: {count}\nPercent: {percent:.1f}%"
        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(fmt_pie_slice(sel.index))
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
        # Table
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        table = ax2.table(cellText=table_data,
                          colLabels=['Category', 'Book'],
                          cellLoc='left',
                          loc='center',
                          colWidths=[0.22, 0.75])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1, 2.2)
        # Style header and first column
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_fontsize(15)
                cell.set_text_props(weight='bold')
                cell.set_height(0.08)
                cell.set_facecolor('#7F0404')
                cell.set_text_props(color='white')
            elif col == 0 and table_data[row-1][0] != '':
                cell.set_facecolor('#FDDE54')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#FFF8E1')
        plt.tight_layout()
        plt.show()

    def program_full_name(self, prog):
        mapping = {
            'BSIT': 'Information Science',
            'BSA': 'Accountancy',
            'BSOA': 'Office Administration',
            'BSHM': 'Hospitality Management',
            'BSCE': 'Civil Engineering',
            'BSEE': 'Electrical Engineering',
            'BSARCHI': 'Architecture',
            'BSAM': 'Applied Mathematics',
            'BSBIO': 'Biology',
            'BSND': 'Nutrition and Dietetics',
            'BEED': 'Elementary Education',
            'BSED': 'Secondary Education',
            'BPA': 'Public Administration',
            'BSBA': 'Business Administration',
            'DCpET': 'Information Science',
            'DEET': 'Electrical Engineering',
            'DCvET': 'Civil Engineering',
            'DICT': 'Information Science',
            'DOMT': 'Office Administration',
        }
        return mapping.get(prog, prog)

    def display_result(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = LibraryApp(root)
    root.mainloop()
