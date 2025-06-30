import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import tkinter as tk
from tkinter import ttk, messagebox
import re
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import mplcursors
import matplotlib.gridspec as gridspec
import os
from PIL import Image, ImageTk  # Requires pillow
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load and preprocess data
DATA_PATH = 'C:/ML_Final_Project/PUP-Library-ML/resources/PUPLibDataset.csv'
df = pd.read_csv(DATA_PATH)

# Ensure df is a pandas DataFrame
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)

def get_student_number_col(df):
    for col in df.columns:
        if col.strip().lower() in ["student number", "studentnumber", "student_no", "studentno"]:
            return col
    return None

student_col_name = get_student_number_col(df)
# Data cleanup and validation: Ensure student numbers are unique to a single program/course
if student_col_name and 'Course and Year' in df.columns:
    # Ensure columns are pandas Series
    student_col = pd.Series(df[student_col_name])
    course_col = pd.Series(df['Course and Year'])
    # Find student numbers used in more than one course/program
    dup_students = df.groupby(student_col_name)['Course and Year'].nunique()
    multi_course_students = list(dup_students[dup_students > 1].index)
    if multi_course_students:
        print(f"Warning: The following student numbers are used in more than one course/program and will be removed for data integrity: {multi_course_students}")
        df = df[~df[student_col_name].isin(multi_course_students)]
    # Optionally, drop duplicate student numbers within the same course/program
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.drop_duplicates(subset=[str(student_col_name), 'Course and Year'])

# Group by Transaction ID to get sets of books borrowed together
transactions = df.groupby('Transaction ID')['Book Title'].apply(list).tolist()

# Create one-hot encoded DataFrame for Apriori
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_ary, columns=pd.Index(te.columns_))

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
        # Place the listbox just below the entry, and lift it above other widgets
        x = self.winfo_x()
        y = self.winfo_y() + self.winfo_height()
        self.lb.place(x=x, y=y, width=self.winfo_width())
        self.lb.lift()
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

class ModernEntry(tk.Entry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(
            font=("Segoe UI", 12),
            bg="#f7f7fa",
            fg="#222",
            relief="flat",
            highlightthickness=2,
            highlightbackground="#e0e0e0",
            highlightcolor="#7F0404",
            bd=0,
            insertbackground="#7F0404"
        )

class ModernButton(tk.Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(
            font=("Segoe UI", 14, "bold"),
            bg="#7F0404",
            fg="white",
            activebackground="#C46B02",
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            height=2,
            width=20
        )

# Tkinter UI
class LibraryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PUP Library Book Recommendation App")
        self.root.geometry('1000x600')
        self.root.resizable(False, False)

        # Save df and rules for stats window
        self.df = df
        self.rules = rules

        # Background image
        bg_path = "C:/ML_Final_Project/PUP-Library-ML/resources/bg.jpg"
        if os.path.exists(bg_path):
            bg_image = Image.open(bg_path)
            bg_image = bg_image.resize((1000, 600), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            bg_label = tk.Label(root, image=self.bg_photo)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            root.configure(bg='#f0f0f0')

        # Modern overlay (rounded corners, shadow, semi-transparent)
        self.shadow = tk.Frame(root, bg='#e0e0e0', bd=0)
        self.shadow.place(relx=0.555, rely=0.085, relwidth=0.4, relheight=0.84)
        self.overlay = tk.Frame(root, bg='#ffffff', bd=0, highlightthickness=0)
        self.overlay.place(relx=0.55, rely=0.08, relwidth=0.4, relheight=0.84)
        self.overlay.config(bg='#ffffff')
        self.overlay.update()
        self.overlay.lift()

        # Logo
        logo_path = "C:/ML_Final_Project/PUP-Library-ML/resources/pup_logo.png"
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path)
            logo_img = logo_img.resize((80, 80), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(self.overlay, image=self.logo_photo, bg='#ffffff')
            logo_label.pack(pady=(20, 10))
        else:
            logo_label = tk.Label(self.overlay, text="[Logo]", font=("Segoe UI", 20, "bold"), bg='#ffffff')
            logo_label.pack(pady=(20, 10))

        # Title
        title = tk.Label(self.overlay, text="PUP LIBRARY", font=("Segoe UI", 26, "bold"), bg='#ffffff', fg='#7F0404')
        title.pack()
        subtitle = tk.Label(self.overlay, text="Recommender System", font=("Segoe UI", 16), bg='#ffffff', fg='#333')
        subtitle.pack(pady=(0, 20))

        # Book Name Entry
        book_label = tk.Label(self.overlay, text="Book Name", font=("Segoe UI", 11), bg='#ffffff', anchor='w')
        book_label.pack(pady=(10, 0), padx=40, anchor='w')
        self.book_entry = AutocompleteEntry(book_titles, self.overlay, width=28, font=("Segoe UI", 12), bd=0, relief='flat', highlightthickness=2, highlightbackground="#e0e0e0", highlightcolor="#7F0404", bg="#f7f7fa", fg="#222", insertbackground="#7F0404")
        self.book_entry.pack(pady=(0, 15), padx=40)

        # Program/Course Autocomplete
        program_label = tk.Label(self.overlay, text="Program", font=("Segoe UI", 11), bg='#ffffff', anchor='w')
        program_label.pack(pady=(0, 0), padx=40, anchor='w')
        self.course_entry = AutocompleteEntry(course_year_list, self.overlay, width=28, font=("Segoe UI", 12), bd=0, relief='flat', highlightthickness=2, highlightbackground="#e0e0e0", highlightcolor="#7F0404", bg="#f7f7fa", fg="#222", insertbackground="#7F0404")
        self.course_entry.pack(pady=(0, 25), padx=40)

        # Recommend Button
        self.recommend_btn = ModernButton(
            self.overlay, text="Recommend", command=self.start_loading
        )
        self.recommend_btn.pack(pady=(10, 0))

        # Results area
        self.result_frame = tk.Frame(self.overlay, bg='#f7f7fa')
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        self.result_text = tk.Text(self.result_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 12), bg='#f7f7fa', relief='flat', bd=0)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Loading animation label (hidden by default)
        self.loading_label = tk.Label(self.overlay, text="", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#7F0404")
        self.loading_label.place(relx=0.5, rely=0.7, anchor="center")
        self.loading_animation_running = False

        # Result panel (hidden by default)
        self.result_panel = tk.Frame(self.root, bg="#fff", bd=0, highlightthickness=0)
        self.result_panel.place(relx=1.0, rely=0.08, relwidth=0.4, relheight=0.84)
        self.result_panel.lower()
        self.result_panel_content = tk.Label(self.result_panel, text="", font=("Segoe UI", 14), bg="#fff", fg="#222", justify="left")
        self.result_panel_content.pack(padx=40, pady=40, anchor="n")

        # Add floating button for book statistics
        self.stats_btn = tk.Button(root, text="📊 Book Stats", font=("Segoe UI", 12, "bold"), bg="#7F0404", fg="white", activebackground="#C46B02", activeforeground="white", bd=0, relief="flat", cursor="hand2", command=self.show_categories_viz)
        self.stats_btn.place(relx=0.97, rely=0.95, anchor="se", width=140, height=40)

        # Show statistics window on app start
        self.root.after(200, lambda: self.show_data_statistics())

    def start_loading(self):
        self.recommend_btn.config(state=tk.DISABLED)
        self.loading_label.config(text="Loading", fg="#7F0404")
        self.loading_label.lift()
        self.loading_animation_running = True
        self.animate_loading(0)
        self.root.after(2000, self.finish_loading)

    def animate_loading(self, step):
        if not self.loading_animation_running:
            return
        dots = '.' * (step % 4)
        self.loading_label.config(text=f"Loading{dots}")
        self.root.after(400, lambda: self.animate_loading(step + 1))

    def finish_loading(self):
        self.loading_animation_running = False
        self.loading_label.config(text="")
        self.recommend_btn.config(state=tk.NORMAL)
        self.animate_panel_slide()

    def animate_panel_slide(self, step=0):
        # Animate overlay panel sliding left, then show result panel
        if step <= 20:
            new_relx = 0.55 - (step * 0.03)
            self.overlay.place_configure(relx=new_relx)
            self.shadow.place_configure(relx=new_relx + 0.005)
            self.root.after(15, lambda: self.animate_panel_slide(step + 1))
        else:
            self.overlay.place_configure(relx=0.0)
            self.shadow.place_configure(relx=0.005)
            self.show_result_panel()

    def show_result_panel(self):
        # Get recommendations and show in result_panel
        book = self.book_entry.get().strip()
        course_year = self.course_entry.get().strip()
        result = self.get_recommendation_text(book, course_year)
        self.result_panel_content.config(text=result)
        self.result_panel.lift()
        self.result_panel.place_configure(relx=0.55)

    def get_recommendation_text(self, book, course_year):
        # This is a refactor of the recommend logic to just return the result string
        if not book or book == "book not available":
            return "Please enter a valid book name."
        if not course_year or course_year == "book not available":
            return "Please enter a valid course and year."
        filtered_tids = transaction_info[transaction_info['Course and Year'].str.upper() == course_year.upper()]['Book Title']
        filtered_transactions = filtered_tids.tolist()
        rules_f = pd.DataFrame()
        if filtered_transactions:
            te_ary_f = te.transform(filtered_transactions)
            df_te_f = pd.DataFrame(te_ary_f, columns=pd.Index(list(te.columns_)))
            frequent_itemsets_f = apriori(df_te_f, min_support=0.01, use_colnames=True)
            rules_f = association_rules(frequent_itemsets_f, metric="confidence", min_threshold=0.1)
            if not isinstance(rules_f, pd.DataFrame):
                rules_f = pd.DataFrame(rules_f)
        if isinstance(rules_f, pd.DataFrame) and not rules_f.empty:
            if 'antecedents' in rules_f.columns:
                rec_rules = rules_f[rules_f['antecedents'].apply(lambda x: book in x)]
                if isinstance(rec_rules, pd.DataFrame) and 'consequents' in rec_rules.columns:
                    rec_rules = rec_rules[rec_rules['consequents'].apply(lambda x: len(x) == 1)]
                else:
                    rec_rules = pd.DataFrame()
            else:
                rec_rules = pd.DataFrame()
        else:
            rec_rules = pd.DataFrame()
        if isinstance(rec_rules, pd.DataFrame) and not rec_rules.empty:
            rec_rules = rec_rules.sort_values(['confidence', 'lift'], ascending=[False, False])
            rec_rules = rec_rules.drop_duplicates(subset='consequents')
            rec_rules = rec_rules.head(3)
            result = ""
            top_recs = []
            for idx, row in enumerate(rec_rules.iterrows(), 1):
                row = row[1]
                consequent_book = next(iter(row['consequents']))
                rec_str = (
                    f"TOP {idx}\n"
                    f"{consequent_book}\n"
                    f"Confidence: {row['confidence']:.2f}\n"
                    f"Support: {row['support']:.2f}\n"
                    f"Lift: {row['lift']:.2f}\n"
                )
                top_recs.append(rec_str)
            while len(top_recs) < 3:
                top_recs.append(f"TOP {len(top_recs)+1}\nNo Recommendation\n")
            result = "\n".join(top_recs)
            return result
        classmates_books = df[df['Course and Year'].str.upper() == course_year.upper()]
        book_series = pd.Series(classmates_books['Book Title'].copy())
        if hasattr(book_series, 'str'):
            mask = book_series.str.lower() != book.lower()
            filtered_books = book_series[mask]
        else:
            filtered_books = book_series
        # Ensure filtered_books is a pandas Series for value_counts
        if not isinstance(filtered_books, pd.Series):
            filtered_books = pd.Series(filtered_books)
        top_books = filtered_books.value_counts().head(3)
        if not top_books.empty:
            result = f"No strong association rules for '{book}'.\nTop 3 books most borrowed by students in {course_year}:\n\n"
            for b, count in top_books.items():
                result += f"Recommended Book: {b}\n  - Times Borrowed: {count}\n\n"
            return result
        return f"No recommendations found for '{book}' in {course_year}."

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
            df_te_prog = pd.DataFrame(te_ary_prog, columns=pd.Index(list(te_prog.columns_)))  # type: ignore
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

    def show_data_statistics(self):
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt

        stats_win = tk.Toplevel(self.root)
        stats_win.title("📊 Library Data & ML Insights")
        stats_win.geometry("900x700")
        stats_win.configure(bg="#f7f7fa")

        # Modern panel background
        panel = tk.Frame(stats_win, bg="#ffffff", bd=0, highlightthickness=0)
        panel.place(relx=0.5, rely=0.05, anchor="n", width=800, height=630)

        # Section: General Summary
        header = tk.Label(panel, text="📚 DATA SUMMARY", font=("Segoe UI", 18, "bold"), bg="#ffffff", fg="#7F0404")
        header.pack(pady=(20, 10), anchor="w", padx=30)

        # Robust summary extraction
        def safe_nunique(col):
            try:
                return pd.Series(self.df[col]).nunique()
            except Exception:
                return "N/A"

        def safe_value_counts(col):
            try:
                return pd.Series(self.df[col]).value_counts()
            except Exception:
                return pd.Series(dtype=int)

        student_col_name = get_student_number_col(self.df)
        if student_col_name:
            total_students = pd.Series(self.df[student_col_name]).nunique()
        else:
            total_students = "N/A"
        total_transactions = len(self.df) if hasattr(self, 'df') else "N/A"
        total_books = safe_nunique('Book Title')
        total_courses = safe_nunique('Course and Year')
        most_borrowed = safe_value_counts('Book Title').idxmax() if not safe_value_counts('Book Title').empty else "N/A"
        most_borrowed_count = safe_value_counts('Book Title').max() if not safe_value_counts('Book Title').empty else "N/A"

        # Two-column grid for summary
        summary_frame = tk.Frame(panel, bg="#ffffff")
        summary_frame.pack(pady=(0, 20), padx=30, anchor="w")
        stats = [
            ("Total Transactions", total_transactions),
            ("Total Unique Students", total_students),
            ("Total Unique Books", total_books),
            ("Total Unique Programs/Courses", total_courses),
            ("Most Borrowed Book", most_borrowed),
            ("Times Borrowed", most_borrowed_count),
        ]
        for i, (label, value) in enumerate(stats):
            tk.Label(summary_frame, text=f"{label}:", font=("Segoe UI", 12, "bold"), bg="#ffffff", anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(summary_frame, text=f"{value}", font=("Segoe UI", 12), bg="#ffffff", anchor="w").grid(row=i, column=1, sticky="w", pady=2, padx=(10,0))

        # Section: Machine Learning Insights
        ml_header = tk.Label(panel, text="🤖 MACHINE LEARNING INSIGHTS (Apriori Algorithm)", font=("Segoe UI", 15, "bold"), bg="#ffffff", fg="#C46B02")
        ml_header.pack(pady=(10, 5), anchor="w", padx=30)

        num_rules = len(self.rules) if hasattr(self, 'rules') else 0
        tk.Label(panel, text=f"Total Association Rules Found: {num_rules}", font=("Segoe UI", 12), bg="#ffffff", anchor="w").pack(anchor="w", padx=40)

        # Top 3 rules by confidence
        if num_rules > 0:
            top_rules = self.rules.sort_values(['confidence', 'lift'], ascending=[False, False]).head(3)
            for idx, row in top_rules.iterrows():
                antecedents = ', '.join(list(row['antecedents']))
                consequents = ', '.join(list(row['consequents']))
                rule_str = (
                    f"Rule: If a student borrows [{antecedents}], they are likely to also borrow [{consequents}]\n"
                    f"  - Confidence: {row['confidence']:.2f}, Support: {row['support']:.2f}, Lift: {row['lift']:.2f}\n"
                )
                tk.Label(panel, text=rule_str, font=("Segoe UI", 11), bg="#ffffff", justify="left", wraplength=700).pack(anchor="w", padx=50)
            # Plot histogram of confidence
            fig, ax = plt.subplots(figsize=(4, 2))
            self.rules['confidence'].plot(kind='hist', bins=10, ax=ax, color="#C46B02")
            ax.set_title("Distribution of Rule Confidence")
            ax.set_xlabel("Confidence")
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=panel)
            canvas.get_tk_widget().pack(pady=10, padx=30, anchor="e")
            canvas.draw()
        else:
            tk.Label(panel, text="No association rules found in the current data.", font=("Segoe UI", 11), fg="red", bg="#ffffff").pack(anchor="w", padx=40)

        # Close button
        tk.Button(panel, text="Close", command=stats_win.destroy, font=("Segoe UI", 11, "bold"), bg="#7F0404", fg="white", bd=0, relief="flat", width=10).pack(side="bottom", pady=18)

if __name__ == "__main__":
    root = tk.Tk()
    app = LibraryApp(root)
    root.mainloop()
