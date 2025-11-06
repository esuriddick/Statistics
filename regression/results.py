#-----------------------------------------------------------------------------#
# ---- MODULES
#-----------------------------------------------------------------------------#
import datetime

#-----------------------------------------------------------------------------#
# ---- FUNCTIONS
#-----------------------------------------------------------------------------#
def summary(model):
    r"""
    Regression Model's Summary
    ===========
    Display a formatted summary of key statistics from a fitted regression model.

    This function prints a concise summary of a regression model, and it is intended for quick inspection
    of model performance and basic metadata, displayed in a neatly formatted console output.

    Parameters
    ----------
    model : object
        A fitted regression model object that must have the following attributes:
            - endog_name : str
                Name of the dependent variable.
            - r_squared : float
                R-squared value of the model.
            - adj_r_squared : float
                Adjusted R-squared value.
            - model_name : str
                Name or description of the model (e.g., 'OLS').

    Returns
    ----------
    None
        This function prints output directly to the console and does not return any value.

    Notes
    -----
    - The output is purely textual and meant for display/logging purposes.
    - Ensure the `model` object has the required attributes, or an `AttributeError` may occur.
    - The formatting assumes a fixed table width for consistent alignment.
    """
    
    # Engine
    #---------------------------------------------------------------------#
    # Auxiliary Variables
    table_width = 78
    column_sep_width = 3
    column_sep = " " * column_sep_width
    column_width = int(table_width / 2 - column_sep_width)
    line_sep_01 = "=" * table_width
    line_sep_02 = "-" * table_width
    
    # Title
    title_label = "OLS Regression Results"
    title = title_label.center(table_width)
    
    # Main - First Row
    text_01 = "Dep. Variable: "
    text_02 = "R-squared: "
    text_var_len_01 = column_width - len(text_01)
    first_row = text_01.ljust(column_width - len(model.endog_name[:text_var_len_01]) + 1, " ")
    first_row += f"{model.endog_name[:(column_width - len(text_01))]}"
    first_row += column_sep
    
    text_var_len_02 = column_width - len(f"{round(model.r_squared, 3):.3f}") + 2
    first_row += text_02.ljust(text_var_len_02, " ")
    first_row += f"{round(model.r_squared, 3):.3f}"
    
    # Main - Second Row
    text_01 = "Model: "
    text_02 = "Adj. R-squared: "
    text_var_len_01 = column_width - len(text_01)
    second_row = text_01.ljust(column_width - len(model.model_name[:text_var_len_01]) + 1, " ")
    second_row += f"{model.model_name[:(column_width - len(text_01))]}"
    second_row += column_sep
    
    text_var_len_02 = column_width - len(f"{round(model.adj_r_squared, 3):.3f}") + 2
    second_row += text_02.ljust(text_var_len_02, " ")
    second_row += f"{round(model.adj_r_squared, 3):.3f}"
    
    # Main - Third Row
    text_01 = "Method: "
    text_02 = "F-statistic: "
    text_var_len_01 = column_width - len(text_01)
    third_row = text_01.ljust(column_width - len(model.method[:text_var_len_01]) + 1, " ")
    third_row += f"{model.method[:(column_width - len(text_01))].upper()}"
    third_row += column_sep
    
    text_var_len_02 = column_width - len(f"{round(model.f_stat, 2):.2f}") + 2
    third_row += text_02.ljust(text_var_len_02, " ")
    third_row += f"{round(model.f_stat, 2):.2f}"
    
    # Main - Fourth Row
    text_01 = "Date (YYYY-MM-DD): "
    text_02 = "Prob (F-statistic): "
    text_var_len_01 = column_width - len(text_01)
    fourth_row = text_01.ljust(column_width - len(str(model.date_fit)) + 1, " ")
    fourth_row += f"{model.date_fit}"
    fourth_row += column_sep
    
    text_var_len_02 = column_width - len(f"{round(model.f_pvalue, 2):.2f}") + 2
    fourth_row += text_02.ljust(text_var_len_02, " ")
    fourth_row += f"{round(model.f_pvalue, 2):.2f}"
    
    # Main - Fifth Row
    text_01 = "Time (HH:MM:SS): "
    text_02 = "Log-Likelihood: "
    text_var_len_01 = column_width - len(text_01)
    fifth_row = text_01.ljust(column_width - len(str(model.time_fit)) + 1, " ")
    fifth_row += f"{model.time_fit}"
    fifth_row += column_sep
    
    text_var_len_02 = column_width - len(f"{round(model.f_pvalue, 2):.2f}") + 2
    fifth_row += text_02.ljust(text_var_len_02, " ")
    fifth_row += f"{round(model.f_pvalue, 2):.2f}"
    
    # Output
    #---------------------------------------------------------------------#
    print(title)
    print(line_sep_01)
    print(first_row)
    print(second_row)
    print(third_row)
    print(fourth_row)
    print(fifth_row)