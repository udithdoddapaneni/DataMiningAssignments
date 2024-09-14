# GiG

from typing import Any

import duckdb
import pandas as pd


def get_fec_dataset(filename: str = "resources/fec_2012_contribution_subset.csv") -> pd.DataFrame:
    # read the csv file into a Pandas data frame

    fec_all = pd.read_csv(filename, low_memory=False)
    # The date is in the format 20-JUN-11
    fec_all["contb_receipt_dt"] = pd.to_datetime(fec_all["contb_receipt_dt"], format="%d-%b-%y")

    # ignore the refunds
    # Get the subset of dataset where contribution amount is positive
    fec_all = fec_all[fec_all.contb_receipt_amt > 0]

    # fec_all contains details about all presidential candidates.
    # fec contains the details about contributions to Barack Obama and Mitt Romney only
    # for the rest of the tasks, unless explicitly specified, work on the fec data frame.
    fec = fec_all[fec_all.cand_nm.isin(["Obama, Barack", "Romney, Mitt"])]

    # Make the original dataset as None so that it will be garbage collected
    fec_all = None

    return fec


def tot_amount_candidate_pandas(fec_df: pd.DataFrame, name: str) -> float:
    return fec_df[fec_df["cand_nm"] == name]["contb_receipt_amt"].sum()


def tot_amount_state_pandas(fec_df: pd.DataFrame, state: str) -> float:
    return fec_df[fec_df["contbr_st"] == state]["contb_receipt_amt"].sum()


def tot_amount_job_pandas(fec_df: pd.DataFrame, candidate: str, company: str, job: str) -> float:
    fec_df = fec_df.dropna(subset=["contb_receipt_amt", "contbr_employer", "contbr_occupation"])
    return fec_df[fec_df["cand_nm"] == candidate][fec_df["contbr_employer"].str.contains(company)][
        fec_df["contbr_occupation"].str.contains(job)
    ]["contb_receipt_amt"].sum()


def tot_contributions_for_cand_pandas(fec_df: pd.DataFrame, candidate: str) -> pd.Series:
    return fec_df[fec_df["cand_nm"] == candidate].groupby("contbr_st")["contbr_nm"].count()


def top_10_state_pandas(fec_df: pd.DataFrame, candidate: str) -> pd.Series:
    return (
        fec_df[fec_df["cand_nm"] == candidate]
        .groupby("contbr_st")["contb_receipt_amt"]
        .count()
        .sort_values(ascending=False)
        .head(n=10)
    )


def discretization_pandas(fec_df: pd.DataFrame) -> pd.DataFrame:
    from copy import deepcopy

    fec_df = deepcopy(fec_df)
    fec_df["bins"] = pd.cut(
        fec_df["contb_receipt_amt"], bins=[0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    )
    ans = fec_df.groupby(["cand_nm", "bins"])["contb_receipt_amt"].sum()
    return ans.unstack(0)


def load_fec_data_to_duckdb(
    filename: str = "resources/fec_2012_contribution_subset.csv",
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")

    con.execute(f"""
        CREATE OR REPLACE TABLE fec_table AS
        SELECT * FROM read_csv_auto('{filename}')
    """)

    return con


def query_fec_data(con: duckdb.DuckDBPyConnection, query: str) -> list[Any]:
    return con.execute(query).fetchall()


t32b1_query = """
select sum(CAST(contb_receipt_amt AS DOUBLE)), cand_nm from fec_table
where (cand_nm = 'Obama, Barack' or cand_nm = 'Romney, Mitt') and
CAST(contb_receipt_amt AS DOUBLE) >= 0
group by cand_nm
order by cand_nm
"""

t32b2_query = """
select count(contbr_nm), contbr_st from fec_table
where (cand_nm = 'Obama, Barack') and contb_receipt_amt > 0
group by contbr_st
order by count(contbr_nm) desc
limit 10
"""


t32c1_query = """
select count(cmte_id), contbr_st from
(select * from read_csv_auto('src/ds5612_pa1/resources/fec_2012_contribution_subset.csv'))
where (cand_nm = 'Obama, Barack') and contb_receipt_amt > 0
group by contbr_st
order by count(cmte_id) desc
limit 10
"""
