from pathlib import Path
import re
import sys
import pandas as pd

# ============================================================
# JOIN FINANCIAL + STOCK PRICE + USD/IDR
# PERIOD: 2023-2024
#
# Expected files in the SAME folder as this script:
#
#   Jan2023.xlsx ... Dec2023.xlsx
#   Jan2024.xlsx ... Dec2024.xlsx
#   compressed_data.csv
#   usd_idr_daily_cleaned.csv
#
# Outputs:
#   financial_2023_2024_all.csv
#   joined_2023_2024_daily.csv
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STOCK_FILE = BASE_DIR / "compressed_data.csv"
FX_FILE = BASE_DIR / "usd_idr_daily_cleaned.csv"

FINANCIAL_OUTPUT = BASE_DIR / "financial_2023_2024_all.csv"
FINAL_OUTPUT = BASE_DIR / "joined_2023_2024_daily.csv"

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

MONTH_ORDER = {name: number for name, number in MONTHS.items()}

FINANCIAL_PATTERN = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(2023|2024)\.xlsx$",
    re.IGNORECASE,
)


# ============================================================
# HELPER
# ============================================================

def normalize_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
    )


def clean_header_value(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


# ============================================================
# READ ONE FINANCIAL EXCEL
# ============================================================

def read_financial_file(path: Path) -> pd.DataFrame:
    print(f"Reading financial: {path.name}")

    raw = pd.read_excel(
        path,
        sheet_name=0,
        header=None,
        engine="openpyxl",
    )

    if raw.shape[0] < 6:
        print(f"WARNING: {path.name} terlalu pendek, dilewati.")
        return pd.DataFrame()

    # Berdasarkan struktur Excel:
    # row index 3 = header level 1
    # row index 4 = header level 2
    header_1 = raw.iloc[3].tolist()
    header_2 = raw.iloc[4].tolist()

    columns = []

    for i, (a, b) in enumerate(zip(header_1, header_2)):
        a = clean_header_value(a)
        b = clean_header_value(b)

        if a and b:
            col = f"{a}_{b}"
        elif a:
            col = a
        elif b:
            col = b
        else:
            col = f"Extra_{i}"

        columns.append(col)

    # Data mulai index 5
    df = raw.iloc[5:].copy()
    df.columns = columns

    # Hanya hapus BARIS yang 100% kosong.
    df = df.dropna(axis=0, how="all")

    # Hanya hapus KOLOM yang 100% kosong.
    df = df.dropna(axis=1, how="all")

    # Buang artificial first column bila benar-benar kosong.
    if len(df.columns) > 0 and df.iloc[:, 0].isna().all():
        df = df.drop(columns=[df.columns[0]])

    # ========================================================
    # IDENTIFIER
    # ========================================================

    if "Code" not in df.columns:
        raise ValueError(
            f"{path.name}: kolom 'Code' tidak ditemukan. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    df["Code"] = normalize_text(df["Code"])

    # Baris tanpa Code dianggap bukan record saham.
    df = df[
        df["Code"].notna()
        & (df["Code"] != "")
    ].copy()

    # ========================================================
    # DATE
    # ========================================================

    if "FS_Date" in df.columns:
        df["FS_Date"] = pd.to_datetime(
            df["FS_Date"],
            errors="coerce",
        )
    elif "FS Date" in df.columns:
        df = df.rename(columns={"FS Date": "FS_Date"})
        df["FS_Date"] = pd.to_datetime(
            df["FS_Date"],
            errors="coerce",
        )

    # ========================================================
    # SOURCE PERIOD
    # ========================================================

    match = FINANCIAL_PATTERN.match(path.name)

    if match:
        month_name = match.group(1).title()
        year = int(match.group(2))

        source_date = pd.Timestamp(
            year=year,
            month=MONTHS[month_name],
            day=1,
        )

        source_period = f"{month_name}{year}"

    else:
        source_date = pd.NaT
        source_period = path.stem

    df["Source_File"] = path.name
    df["Source_Period"] = source_period
    df["Source_Date"] = source_date

    return df


# ============================================================
# READ ALL FINANCIAL FILES
# ============================================================

def load_all_financial() -> pd.DataFrame:

    files = [
        p
        for p in BASE_DIR.iterdir()
        if p.is_file() and FINANCIAL_PATTERN.match(p.name)
    ]

    files.sort(
        key=lambda p: (
            int(p.stem[-4:]),
            MONTHS[p.stem[:3].title()],
        )
    )

    print()
    print("========================================")
    print("FINANCIAL FILES")
    print("========================================")
    print(f"Financial files ditemukan: {len(files)}")

    if not files:
        raise FileNotFoundError(
            "Tidak ada file Jan2023.xlsx sampai Dec2024.xlsx "
            "di folder yang sama dengan script."
        )

    frames = []

    for path in files:
        df = read_financial_file(path)

        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(
            "Tidak ada financial data yang berhasil dibaca."
        )

    # OUTER concat:
    # semua kolom dari semua file dipertahankan.
    financial = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    # Buang hanya kolom yang seluruhnya kosong.
    financial = financial.dropna(
        axis=1,
        how="all",
    )

    # Rename Code -> Ticker untuk proses join.
    financial = financial.rename(
        columns={"Code": "Ticker"}
    )

    financial["Ticker"] = normalize_text(
        financial["Ticker"]
    )

    print(f"Financial rows : {len(financial):,}")
    print(f"Financial cols : {len(financial.columns):,}")

    return financial


# ============================================================
# READ STOCK PRICE
# ============================================================

def load_stock() -> pd.DataFrame:

    if not STOCK_FILE.exists():

        raise FileNotFoundError(
            f"Tidak menemukan:\n{STOCK_FILE}"
        )

    print()
    print("========================================")
    print("STOCK PRICE")
    print("========================================")
    print(f"File: {STOCK_FILE.name}")

    stock = pd.read_csv(
        STOCK_FILE,
        low_memory=False,
    )

    # Beberapa dataset memakai lowercase.
    rename = {}
    for col in stock.columns:
        normalized = str(col).strip().lower()

        if normalized == "date":
            rename[col] = "Date"
        elif normalized == "ticker":
            rename[col] = "Ticker"

    stock = stock.rename(columns=rename)

    required = {"Date", "Ticker"}
    missing = required - set(stock.columns)

    if missing:
        raise ValueError(
            "Kolom wajib stock tidak ditemukan: "
            f"{sorted(missing)}\n"
            f"Kolom tersedia: {list(stock.columns)}"
        )

    stock["Date"] = pd.to_datetime(
        stock["Date"],
        errors="coerce",
    )

    stock["Ticker"] = normalize_text(
        stock["Ticker"]
    )

    # Hanya tahun 2023-2024.
    stock = stock[
        stock["Date"].dt.year.isin([2023, 2024])
    ].copy()

    # Jangan buang row hanya karena ada nilai OHLC/Volume kosong.
    # Buang hanya Date/Ticker yang tidak valid.
    stock = stock.dropna(
        subset=["Date", "Ticker"]
    ).copy()

    # Duplicate exact ticker-date:
    # simpan record terakhir.
    stock = (
        stock
        .sort_values(
            ["Date", "Ticker"],
            kind="stable",
        )
        .drop_duplicates(
            subset=["Date", "Ticker"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(f"Stock rows   : {len(stock):,}")
    print(f"Stock tickers: {stock['Ticker'].nunique():,}")

    return stock


# ============================================================
# READ USD/IDR
# ============================================================

def load_fx() -> pd.DataFrame:

    if not FX_FILE.exists():

        raise FileNotFoundError(
            f"Tidak menemukan:\n{FX_FILE}"
        )

    print()
    print("========================================")
    print("USD/IDR")
    print("========================================")
    print(f"File: {FX_FILE.name}")

    fx = pd.read_csv(
        FX_FILE,
        low_memory=False,
    )

    # Normalize Date column name.
    date_col = None

    for col in fx.columns:
        if str(col).strip().lower() == "date":
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            "usd_idr_daily_cleaned.csv tidak punya kolom Date."
        )

    if date_col != "Date":
        fx = fx.rename(columns={date_col: "Date"})

    # Parse Date tanpa timezone problem.
    fx["Date"] = (
        pd.to_datetime(
            fx["Date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    fx = fx.dropna(
        subset=["Date"]
    ).copy()

    fx = fx[
        fx["Date"].dt.year.isin([2023, 2024])
    ].copy()

    # Hapus duplicate tanggal.
    fx = (
        fx
        .sort_values("Date", kind="stable")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Rename semua kolom selain Date.
    rename_fx = {}

    for col in fx.columns:
        if col != "Date":
            rename_fx[col] = f"USD_IDR_{col}"

    fx = fx.rename(
        columns=rename_fx
    )

    print(f"USD/IDR rows: {len(fx):,}")

    return fx


# ============================================================
# JOIN FINANCIAL SNAPSHOT -> DAILY STOCK
#
# Kenapa tidak langsung merge_asof sekali?
# Karena pandas merge_asof bisa menghasilkan:
# "left keys must be sorted"
# ketika digabung dengan `by`.
#
# Solusi robust:
# lakukan merge_asof PER TICKER.
# ============================================================

def join_financial_to_stock(
    financial: pd.DataFrame,
    stock: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Joining financial -> daily stock...")

    # Financial harus punya Source_Date.
    financial["Source_Date"] = pd.to_datetime(
        financial["Source_Date"],
        errors="coerce",
    )

    financial = financial.dropna(
        subset=["Ticker", "Source_Date"]
    ).copy()

    # Hanya data 2023-2024.
    financial = financial[
        financial["Source_Date"].dt.year.isin([2023, 2024])
    ].copy()

    # Hapus duplicate snapshot ticker + Source_Date.
    financial = (
        financial
        .sort_values(
            ["Ticker", "Source_Date"],
            kind="stable",
        )
        .drop_duplicates(
            subset=["Ticker", "Source_Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    stock = stock.copy()

    # Pisahkan ticker yang tersedia di kedua dataset.
    stock_tickers = set(
        stock["Ticker"].dropna().unique()
    )

    financial_tickers = set(
        financial["Ticker"].dropna().unique()
    )

    common_tickers = stock_tickers & financial_tickers

    print(
        f"Common tickers: {len(common_tickers):,}"
    )

    result_frames = []

    all_financial_columns = [
        c
        for c in financial.columns
        if c not in {"Ticker", "Source_Date"}
    ]

    # Ticker satu per satu -> merge_asof aman.
    for ticker, stock_group in stock.groupby(
        "Ticker",
        sort=False,
    ):

        stock_group = stock_group.sort_values(
            "Date",
            kind="stable",
        ).reset_index(drop=True)

        financial_group = financial[
            financial["Ticker"] == ticker
        ].copy()

        # Tidak ada financial snapshot untuk ticker ini.
        if financial_group.empty:

            for col in all_financial_columns:

                if col not in stock_group.columns:
                    stock_group[col] = pd.NA

            stock_group["Source_Date"] = pd.NaT

            result_frames.append(
                stock_group
            )

            continue

        financial_group = financial_group.sort_values(
            "Source_Date",
            kind="stable",
        ).reset_index(drop=True)

        # Tanggal stock:
        # ambil snapshot financial TERAKHIR yang <= tanggal stock.
        merged = pd.merge_asof(
            stock_group,
            financial_group,
            left_on="Date",
            right_on="Source_Date",
            direction="backward",
            allow_exact_matches=True,
            suffixes=("", "_financial"),
        )

        result_frames.append(
            merged
        )

    joined = pd.concat(
        result_frames,
        ignore_index=True,
        sort=False,
    )

    return joined


# ============================================================
# JOIN USD/IDR BY DATE
# ============================================================

def join_usd_idr(
    joined: pd.DataFrame,
    fx: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("Joining USD/IDR -> Date...")

    joined["Date"] = pd.to_datetime(
        joined["Date"],
        errors="coerce",
    )

    fx["Date"] = pd.to_datetime(
        fx["Date"],
        errors="coerce",
    )

    # Exact daily match.
    joined = joined.merge(
        fx,
        how="left",
        on="Date",
        sort=False,
        validate="many_to_one",
    )

    return joined


# ============================================================
# FINAL CLEAN + ORDER
# ============================================================

def final_cleanup(
    joined: pd.DataFrame,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # HANYA HAPUS COLUMN 100% KOSONG
    # --------------------------------------------------------

    joined = joined.dropna(
        axis=1,
        how="all",
    )

    # --------------------------------------------------------
    # SORT FINAL
    #
    # 1. Date ascending
    # 2. Ticker ascending
    # --------------------------------------------------------

    joined["Date"] = pd.to_datetime(
        joined["Date"],
        errors="coerce",
    )

    joined = (
        joined
        .sort_values(
            ["Date", "Ticker"],
            ascending=[True, True],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # NO HARUS 1,2,3,...N
    # --------------------------------------------------------

    if "No" in joined.columns:
        joined = joined.drop(
            columns=["No"]
        )

    joined.insert(
        0,
        "No",
        range(
            1,
            len(joined) + 1,
        ),
    )

    # --------------------------------------------------------
    # COLUMN ORDER
    #
    # No
    # Date
    # Ticker
    # stock columns
    # financial columns
    # USD/IDR columns
    # --------------------------------------------------------

    preferred_first = [
        "No",
        "Date",
        "Ticker",
    ]

    preferred_last = [
        "Source_File",
        "Source_Period",
        "Source_Date",
    ]

    usd_columns = [
        c
        for c in joined.columns
        if c.startswith("USD_IDR_")
    ]

    other_columns = [
        c
        for c in joined.columns
        if c not in preferred_first
        and c not in preferred_last
        and c not in usd_columns
    ]

    final_columns = (
        [c for c in preferred_first if c in joined.columns]
        + other_columns
        + [c for c in preferred_last if c in joined.columns]
        + usd_columns
    )

    # Remove accidental duplicates while preserving order.
    final_columns = list(
        dict.fromkeys(final_columns)
    )

    joined = joined[
        [
            c
            for c in final_columns
            if c in joined.columns
        ]
    ]

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    expected_no = pd.Series(
        range(1, len(joined) + 1),
        index=joined.index,
        dtype="int64",
    )

    if not joined["No"].equals(expected_no):
        raise ValueError(
            "VALIDATION ERROR: No tidak berurutan 1..N."
        )

    if not joined["Date"].is_monotonic_increasing:
        raise ValueError(
            "VALIDATION ERROR: Date tidak ascending."
        )

    if joined.duplicated(
        subset=["Date", "Ticker"]
    ).any():

        duplicated_count = int(
            joined.duplicated(
                subset=["Date", "Ticker"]
            ).sum()
        )

        print(
            f"WARNING: masih ada {duplicated_count:,} "
            "duplicate Date+Ticker."
        )

    return joined


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    financial: pd.DataFrame,
    joined: pd.DataFrame,
):

    # Financial standalone.
    financial = financial.copy()

    # Financial ordering.
    financial = financial.sort_values(
        ["Source_Date", "Ticker"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    # Financial No.
    if "No" in financial.columns:
        financial = financial.drop(
            columns=["No"]
        )

    financial.insert(
        0,
        "No",
        range(
            1,
            len(financial) + 1,
        ),
    )

    financial.to_csv(
        FINANCIAL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )

    joined.to_csv(
        FINAL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print("JOIN FINANCIAL + STOCK PRICE + USD/IDR")
    print("2023 - 2024")
    print("==============================================")
    print(f"Folder: {BASE_DIR}")
    print()

    # --------------------------------------------------------
    # 1. Financial
    # --------------------------------------------------------

    financial = load_all_financial()

    # --------------------------------------------------------
    # 2. Stock
    # --------------------------------------------------------

    stock = load_stock()

    # --------------------------------------------------------
    # 3. USD/IDR
    # --------------------------------------------------------

    fx = load_fx()

    # --------------------------------------------------------
    # 4. Financial -> Stock
    # --------------------------------------------------------

    joined = join_financial_to_stock(
        financial=financial,
        stock=stock,
    )

    # --------------------------------------------------------
    # 5. USD/IDR -> Joined
    # --------------------------------------------------------

    joined = join_usd_idr(
        joined=joined,
        fx=fx,
    )

    # --------------------------------------------------------
    # 6. Cleanup + order + No
    # --------------------------------------------------------

    joined = final_cleanup(
        joined
    )

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    save_csv(
        financial=financial,
        joined=joined,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("JOIN SELESAI")
    print("==============================================")

    print(
        f"Financial rows : {len(financial):,}"
    )

    print(
        f"Stock rows     : {len(stock):,}"
    )

    print(
        f"USD/IDR rows   : {len(fx):,}"
    )

    print(
        f"FINAL rows     : {len(joined):,}"
    )

    print(
        f"FINAL columns  : {len(joined.columns):,}"
    )

    print()
    print(
        "Tanggal final  : "
        f"{joined['Date'].min()} -> {joined['Date'].max()}"
    )

    print(
        "Ticker final   : "
        f"{joined['Ticker'].nunique():,}"
    )

    print()
    print("Output 1:")
    print(FINANCIAL_OUTPUT)

    print()
    print("Output 2:")
    print(FINAL_OUTPUT)

    print()
    print("5 baris pertama:")
    print(
        joined.head(5).to_string(index=False)
    )

    print()
    print("5 baris terakhir:")
    print(
        joined.tail(5).to_string(index=False)
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nProgram dihentikan.")
        sys.exit(1)

    except Exception as exc:
        print()
        print("==============================================")
        print("ERROR")
        print("==============================================")
        print(exc)
        sys.exit(1)
