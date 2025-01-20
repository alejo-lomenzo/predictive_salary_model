import pytest
from project_pwc.dataset import load_dataframes, merge_data, fill_missing_values

def test_load_dataframes() -> None:

    salary_df, people_df, desc_df = load_dataframes()
    assert not salary_df.empty, "salary_df está vacío!"
    assert not people_df.empty, "people_df está vacío!"
    assert not desc_df.empty, "desc_df está vacío!"

def test_merge_data() -> None:

    salary_df, people_df, desc_df = load_dataframes()
    df_merged = merge_data(salary_df, people_df, desc_df)
    assert "Salary" in df_merged.columns, "No se encuentra la columna 'Salary' tras el merge."
    assert "Description" in df_merged.columns, "No se encuentra la columna 'Description' tras el merge."
    assert df_merged.shape[0] > 0, "El dataframe merged está vacío!"

def test_fill_missing_values() -> None:

    salary_df, people_df, desc_df = load_dataframes()
    df_merged = merge_data(salary_df, people_df, desc_df)
    df_filled = fill_missing_values(df_merged)
    assert df_filled["Salary"].notnull().all(), "Existen nulos en 'Salary' tras fill_missing_values."
    assert df_filled["Age"].dtype == int, "La columna 'Age' no es int."
    assert df_filled["Years of Experience"].dtype == int, "La columna 'Years of Experience' no es int."
