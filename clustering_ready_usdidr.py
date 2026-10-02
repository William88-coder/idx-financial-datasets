from pathlib import Path
import re
import sys
import pandas as pd
import numpy as np

# ============================================================
# CLUSTERING-READY DATASET
# ============================================================
#
# INPUT:
#   Jan2023.xlsx ... Dec2023.xlsx
#   Jan2024.xlsx ... Dec2024.xlsx
#   compressed_data.csv
#   usd_idr_daily_cleaned.csv
#
# OUTPUT:
#   clustering_ready_2023_2024.csv
#
# HASIL:
#   - 1 BARIS = 1 TICKER
#   - Tidak ada data harian berulang
#   - Harga saham tetap ada dalam bentuk fitur
#   - Financial tetap ada dalam bentuk fitur
#   - USD/IDR tetap ada dalam bentuk fitur
#   - Hanya 1 kolom No
#   - No = 1..N
#   - Ticker tidak duplicate
#   - Kolom terlalu banyak NULL dibuang
#   - NULL numeric yang tersisa diisi median
#
# Catatan penting:
#   Data saham yang tersedia menentukan periode harga.
#   Financial snapshot yang dipakai adalah snapshot TERAKHIR
#   yang tersedia sampai tanggal terakhir data saham,
#   sehingga tidak menggunakan data financial dari masa depan.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STOCK_FILE = BASE_DIR / "compressed_data.csv"
FX_FILE = BASE_DIR / "usd_idr_daily_cleaned.csv"

OUTPUT = BASE_DIR / "clustering_ready_2023_2024.csv"

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

FILE_PATTERN = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(2023|2024)\.xlsx$",
    re.IGNORECASE,
)

# Financial features yang dipertahankan.
FINANCIAL_FEATURES = [
    "Assets_bIDR",
    "Liabilities_bIDR",
    "Equity_bIDR",
    "Sales_bIDR",
    "EBT_bIDR",
    "Profit_for_Period",
    "Profit_attributable_to_owners",
    "EPS_IDR",
    "Book_Value_IDR",
    "PE_Ratio_x",
    "Price_to_BV_x",
    "DE_Ratio_x",
    "ROA_pct",
    "ROE_pct",
    "NPM_pct",
]

# Buang kolom jika >50% nilainya kosong.
MAX_COLUMN_MISSING = 0.50

# Satu ticker harus memiliki minimal 60% numeric feature yang tersedia.
MIN_ROW_COMPLETENESS = 0.60


# ============================================================
# HELPERS
# ============================================================

def clean_header(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_ticker(series):
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
    )


def to_numeric(series):
    """Konversi angka/string menjadi numeric dengan aman."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype("string").str.strip()

    s = s.replace({
        "": pd.NA,
        "-": pd.NA,
        "—": pd.NA,
        "–": pd.NA,
        "N/A": pd.NA,
        "NA": pd.NA,
        "null": pd.NA,
        "None": pd.NA,
    })

    # (123.45) -> -123.45
    s = s.str.replace(
        r"^\((.*)\)$",
        r"-\1",
        regex=True
    )

    # comma ribuan
    s = s.str.replace(",", "", regex=False)

    # persen
    s = s.str.replace("%", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


# ============================================================
# FINANCIAL - BACA 24 EXCEL
# ============================================================

def read_financial_file(path):

    raw = pd.read_excel(
        path,
        sheet_name=0,
        header=None,
        engine="openpyxl",
    )

    if raw.shape[0] < 6:
        return pd.DataFrame()

    # Header sumber:
    # row index 3 + row index 4
    h1 = raw.iloc[3].tolist()
    h2 = raw.iloc[4].tolist()

    columns = []

    for i, (a, b) in enumerate(zip(h1, h2)):

        a = clean_header(a)
        b = clean_header(b)

        if a and b:
            name = f"{a}_{b}"
        elif a:
            name = a
        elif b:
            name = b
        else:
            name = f"Extra_{i}"

        columns.append(name)

    # Data mulai index 5.
    df = raw.iloc[5:].copy()
    df.columns = columns

    # Hapus hanya row/column yang 100% kosong.
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Cari Code.
    code_col = None

    for col in df.columns:

        if str(col).strip().lower() == "code":
            code_col = col
            break

    if code_col is None:
        return pd.DataFrame()

    df["Ticker"] = normalize_ticker(
        df[code_col]
    )

    # Hanya row dengan ticker valid.
    df = df[
        df["Ticker"].notna()
        & (df["Ticker"] != "")
    ].copy()

    # --------------------------------------------------------
    # FS Date
    # --------------------------------------------------------

    fs_col = None

    for col in df.columns:

        normalized = (
            str(col)
            .strip()
            .lower()
            .replace("_", " ")
        )

        if normalized == "fs date":
            fs_col = col
            break

    if fs_col is not None:
        df["FS_Date"] = pd.to_datetime(
            df[fs_col],
            errors="coerce",
        )
    else:
        df["FS_Date"] = pd.NaT

    # --------------------------------------------------------
    # Source period/date
    # --------------------------------------------------------

    match = FILE_PATTERN.match(path.name)

    if match:

        month = match.group(1).title()
        year = int(match.group(2))

        df["Source_Date"] = pd.Timestamp(
            year=year,
            month=MONTHS[month],
            day=1,
        )

        df["Source_Period"] = f"{month}{year}"

    else:

        df["Source_Date"] = pd.NaT
        df["Source_Period"] = path.stem

    # --------------------------------------------------------
    # Nama feature
    # --------------------------------------------------------

    rename_map = {}

    for col in df.columns:

        c = str(col).strip().lower()

        if "assets" in c and "b.idr" in c:
            rename_map[col] = "Assets_bIDR"

        elif "liabilities" in c and "b.idr" in c:
            rename_map[col] = "Liabilities_bIDR"

        elif "equity" in c and "b.idr" in c:
            rename_map[col] = "Equity_bIDR"

        elif "sales" in c and "b.idr" in c:
            rename_map[col] = "Sales_bIDR"

        elif "ebt" in c and "b.idr" in c:
            rename_map[col] = "EBT_bIDR"

        elif c == "profit for the period":
            rename_map[col] = "Profit_for_Period"

        elif "profit attr.to owner's" in c:
            rename_map[col] = "Profit_attributable_to_owners"

        elif "trailing_eps" in c:
            rename_map[col] = "EPS_IDR"

        elif "book value" in c:
            rename_map[col] = "Book_Value_IDR"

        elif "p/e ratio" in c:
            rename_map[col] = "PE_Ratio_x"

        elif "price to bv" in c:
            rename_map[col] = "Price_to_BV_x"

        elif "d/e ratio" in c:
            rename_map[col] = "DE_Ratio_x"

        elif "roa" in c:
            rename_map[col] = "ROA_pct"

        elif "roe" in c:
            rename_map[col] = "ROE_pct"

        elif "npm" in c:
            rename_map[col] = "NPM_pct"

    df = df.rename(columns=rename_map)

    # Hanya ambil feature yang diperlukan.
    keep = [
        "Ticker",
        "FS_Date",
        "Source_Date",
        "Source_Period",
    ]

    for feature in FINANCIAL_FEATURES:

        if feature in df.columns:

            df[feature] = to_numeric(
                df[feature]
            )

            keep.append(feature)

    # Metadata.
    for col in ["Sector", "Sharia"]:

        if col in df.columns:
            keep.append(col)

    keep = list(dict.fromkeys(keep))

    return df[
        [c for c in keep if c in df.columns]
    ].copy()


def load_financial():

    files = [
        p
        for p in BASE_DIR.iterdir()
        if p.is_file() and FILE_PATTERN.match(p.name)
    ]

    files.sort(
        key=lambda p: (
            int(p.stem[-4:]),
            MONTHS[p.stem[:3].title()],
        )
    )

    print()
    print("FINANCIAL FILES")
    print("----------------------------------------")
    print(f"Files ditemukan: {len(files)}")

    if len(files) != 24:

        print(
            f"WARNING: seharusnya 24 file untuk Jan2023-Dec2024, "
            f"tetapi ditemukan {len(files)}."
        )

    if not files:

        raise FileNotFoundError(
            "Tidak menemukan Jan2023.xlsx ... Dec2024.xlsx."
        )

    frames = []

    for path in files:

        print(f"Reading: {path.name}")

        df = read_financial_file(path)

        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(
            "Financial tidak menghasilkan data."
        )

    financial = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    financial = financial[
        financial["Source_Date"]
        .dt.year
        .isin([2023, 2024])
    ].copy()

    print(
        f"Financial rows: {len(financial):,}"
    )

    return financial


# ============================================================
# FINANCIAL FEATURE:
# PILIH SNAPSHOT TERAKHIR YANG TIDAK MELEWATI
# TANGGAL TERAKHIR DATA STOCK
# ============================================================

def build_financial_features(
    financial,
    stock_last_date,
):

    financial = financial.dropna(
        subset=["Ticker", "Source_Date"]
    ).copy()

    # Jangan gunakan financial masa depan.
    financial = financial[
        financial["Source_Date"] <= stock_last_date
    ].copy()

    financial = financial.sort_values(
        ["Ticker", "Source_Date"],
        kind="stable"
    )

    # Satu ticker -> satu snapshot terbaru.
    latest = (
        financial
        .groupby(
            "Ticker",
            as_index=False,
            sort=False
        )
        .tail(1)
        .copy()
    )

    keep = ["Ticker"]

    # Metadata.
    for col in ["Sector", "Sharia"]:
        if col in latest.columns:
            keep.append(col)

    # Financial feature.
    for col in FINANCIAL_FEATURES:
        if col in latest.columns:
            keep.append(col)

    result = latest[
        list(dict.fromkeys(keep))
    ].copy()

    return result


# ============================================================
# STOCK PRICE
# ============================================================

def load_stock():

    if STOCK_FILE.exists():

        stock_path = STOCK_FILE

    else:

        fallback = BASE_DIR / "compressed_data(1).csv"

        if fallback.exists():
            stock_path = fallback
        else:
            raise FileNotFoundError(
                "Tidak menemukan compressed_data.csv"
            )

    print()
    print("STOCK PRICE")
    print("----------------------------------------")
    print(f"File: {stock_path.name}")

    stock = pd.read_csv(
        stock_path,
        low_memory=False
    )

    rename = {}

    for col in stock.columns:

        low = str(col).strip().lower()

        if low == "date":
            rename[col] = "Date"

        elif low == "ticker":
            rename[col] = "Ticker"

        elif low == "open":
            rename[col] = "Open"

        elif low == "high":
            rename[col] = "High"

        elif low == "low":
            rename[col] = "Low"

        elif low == "close":
            rename[col] = "Close"

        elif low == "adj close":
            rename[col] = "Adj Close"

        elif low == "volume":
            rename[col] = "Volume"

    stock = stock.rename(
        columns=rename
    )

    required = {
        "Date",
        "Ticker",
        "Close",
    }

    missing = required - set(stock.columns)

    if missing:

        raise ValueError(
            f"Kolom stock wajib hilang: {sorted(missing)}"
        )

    stock["Date"] = pd.to_datetime(
        stock["Date"],
        errors="coerce"
    )

    stock["Ticker"] = normalize_ticker(
        stock["Ticker"]
    )

    # Hanya 2023-2024.
    stock = stock[
        stock["Date"].dt.year.isin([2023, 2024])
    ].copy()

    # Jangan buang row hanya karena OHLC/Volume kosong.
    # Buang hanya Date/Ticker/Close yang tidak valid.
    stock = stock.dropna(
        subset=[
            "Date",
            "Ticker",
            "Close"
        ]
    ).copy()

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]:

        if col in stock.columns:
            stock[col] = to_numeric(
                stock[col]
            )

    # Exact duplicate Date + Ticker.
    stock = (
        stock
        .sort_values(
            ["Ticker", "Date"],
            kind="stable"
        )
        .drop_duplicates(
            subset=["Ticker", "Date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    print(
        f"Stock rows: {len(stock):,}"
    )

    print(
        f"Stock tickers: {stock['Ticker'].nunique():,}"
    )

    print(
        f"Stock period: {stock['Date'].min().date()} "
        f"-> {stock['Date'].max().date()}"
    )

    return stock


# ============================================================
# STOCK -> 1 BARIS PER TICKER
# ============================================================

def max_drawdown(close):

    running_max = close.cummax()

    drawdown = (
        close / running_max - 1
    )

    return drawdown.min() * 100


def build_stock_features(stock):

    stock = stock.sort_values(
        ["Ticker", "Date"],
        kind="stable"
    ).copy()

    # Return harian.
    stock["Daily_Return"] = (
        stock
        .groupby("Ticker")["Close"]
        .pct_change()
    )

    # Daily range.
    if (
        "High" in stock.columns
        and "Low" in stock.columns
    ):

        stock["Daily_Range_Pct"] = (
            (stock["High"] - stock["Low"])
            / stock["Close"]
        ) * 100

    rows = []

    for ticker, group in stock.groupby(
        "Ticker",
        sort=False
    ):

        g = group.sort_values(
            "Date"
        ).copy()

        close = g["Close"].dropna()

        if close.empty:
            continue

        start_close = close.iloc[0]
        end_close = close.iloc[-1]

        total_return = np.nan

        if start_close != 0:

            total_return = (
                (end_close / start_close - 1)
                * 100
            )

        row = {
            "Ticker": ticker,

            # ----------------------------
            # HARGA SAHAM
            # ----------------------------

            "Stock_Start_Close": start_close,
            "Stock_End_Close": end_close,
            "Stock_Avg_Close": close.mean(),
            "Stock_Min_Close": close.min(),
            "Stock_Max_Close": close.max(),

            # Return.
            "Stock_Total_Return_pct": total_return,

            # Return harian.
            "Stock_Avg_Daily_Return_pct": (
                g["Daily_Return"].mean() * 100
            ),

            # Volatility.
            "Stock_Volatility_pct": (
                g["Daily_Return"].std() * 100
            ),

            # Maximum drawdown.
            "Stock_Max_Drawdown_pct": (
                max_drawdown(close)
            ),
        }

        if "Volume" in g.columns:

            row["Stock_Avg_Volume"] = (
                g["Volume"].mean()
            )

            row["Stock_Total_Volume"] = (
                g["Volume"].sum()
            )

        if "Daily_Range_Pct" in g.columns:

            row["Stock_Avg_Daily_Range_pct"] = (
                g["Daily_Range_Pct"].mean()
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    # Harus satu row per ticker.
    result = (
        result
        .sort_values("Ticker")
        .drop_duplicates("Ticker", keep="last")
        .reset_index(drop=True)
    )

    print()
    print(
        f"Stock feature rows: {len(result):,}"
    )

    return result


# ============================================================
# USD/IDR
# ============================================================

def load_fx():

    if not FX_FILE.exists():

        raise FileNotFoundError(
            "Tidak menemukan usd_idr_daily_cleaned.csv"
        )

    print()
    print("USD/IDR")
    print("----------------------------------------")
    print(
        f"File: {FX_FILE.name}"
    )

    fx = pd.read_csv(
        FX_FILE,
        low_memory=False
    )

    rename = {}

    for col in fx.columns:

        low = str(col).strip().lower()

        if low == "date":
            rename[col] = "Date"

        elif low == "open":
            rename[col] = "Open"

        elif low == "high":
            rename[col] = "High"

        elif low == "low":
            rename[col] = "Low"

        elif low == "close":
            rename[col] = "Close"

    fx = fx.rename(
        columns=rename
    )

    if (
        "Date" not in fx.columns
        or "Close" not in fx.columns
    ):

        raise ValueError(
            "USD/IDR harus mempunyai Date dan Close."
        )

    fx["Date"] = (
        pd.to_datetime(
            fx["Date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    fx = fx[
        fx["Date"].dt.year.isin([2023, 2024])
    ].copy()

    fx = fx.dropna(
        subset=["Date", "Close"]
    ).copy()

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        if col in fx.columns:
            fx[col] = to_numeric(
                fx[col]
            )

    fx = (
        fx
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    print(
        f"USD/IDR rows: {len(fx):,}"
    )

    print(
        f"USD/IDR period: "
        f"{fx['Date'].min().date()} "
        f"-> {fx['Date'].max().date()}"
    )

    return fx


def build_fx_features(fx):

    fx = fx.sort_values(
        "Date"
    ).copy()

    start_close = fx["Close"].iloc[0]
    end_close = fx["Close"].iloc[-1]

    return pd.DataFrame([{
        "USD_IDR_Start_Close": start_close,
        "USD_IDR_End_Close": end_close,
        "USD_IDR_Avg_Close": fx["Close"].mean(),
        "USD_IDR_Min_Close": fx["Close"].min(),
        "USD_IDR_Max_Close": fx["Close"].max(),

        "USD_IDR_Total_Change_pct": (
            (
                end_close / start_close - 1
            ) * 100
            if start_close != 0
            else np.nan
        )
    }])


# ============================================================
# BUILD FINAL
# ============================================================

def build_final(
    financial,
    stock,
    stock_features,
    fx_features,
):

    # --------------------------------------------------------
    # Financial latest snapshot
    # TIDAK BOLEH lebih baru daripada akhir stock period.
    # --------------------------------------------------------

    stock_last_date = stock["Date"].max()

    fin_features = build_financial_features(
        financial,
        stock_last_date=stock_last_date,
    )

    # --------------------------------------------------------
    # JOIN STOCK + FINANCIAL
    # --------------------------------------------------------

    final = stock_features.merge(
        fin_features,
        how="left",
        on="Ticker",
        validate="one_to_one",
        suffixes=("", "_financial")
    )

    # --------------------------------------------------------
    # JOIN USD/IDR
    #
    # FX hanya 1 row.
    # Final punya banyak ticker.
    # Jadi validation yang benar:
    # MANY_TO_ONE, bukan ONE_TO_ONE.
    # --------------------------------------------------------

    final = final.merge(
        fx_features,
        how="left",
        left_index=True,
        right_index=True,
        validate="many_to_one",
        suffixes=("", "_fx")
    )

    # --------------------------------------------------------
    # HAPUS DATE/MONTH/FILE METADATA
    #
    # Karena output ini khusus untuk clustering.
    # Periode data sudah 2023-2024 sesuai input.
    # --------------------------------------------------------

    date_metadata = [
        col
        for col in [
            "Date",
            "Price_Start_Date",
            "Price_End_Date",
            "FS_Date",
            "Source_Date",
            "Source_Period",
            "Financial_FS_Date",
            "Financial_Source_Date",
            "Source_File",
        ]
        if col in final.columns
    ]

    if date_metadata:
        final = final.drop(
            columns=date_metadata
        )

    # --------------------------------------------------------
    # DUPLICATE No:
    # HAPUS SEMUA No DULU.
    # NANTI DIBUAT SATU KALI.
    # --------------------------------------------------------

    no_columns = [
        col
        for col in final.columns
        if str(col).strip().lower() == "no"
    ]

    if no_columns:
        final = final.drop(
            columns=no_columns
        )

    # --------------------------------------------------------
    # HAPUS KOLOM 100% KOSONG
    # --------------------------------------------------------

    final = final.dropna(
        axis=1,
        how="all"
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    if "Sector" not in final.columns:
        final["Sector"] = "Unknown"

    if "Sharia" not in final.columns:
        final["Sharia"] = "Unknown"

    final["Sector"] = (
        final["Sector"]
        .fillna("Unknown")
        .astype("string")
        .str.strip()
    )

    final["Sharia"] = (
        final["Sharia"]
        .fillna("Unknown")
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    id_columns = {
        "Ticker",
        "Sector",
        "Sharia",
    }

    numeric_columns = [
        c
        for c in final.columns
        if c not in id_columns
        and pd.api.types.is_numeric_dtype(
            final[c]
        )
    ]

    # --------------------------------------------------------
    # HAPUS NUMERIC FEATURE YANG TIDAK BERINFORMASI
    # --------------------------------------------------------

    keep_numeric = []

    for col in numeric_columns:

        non_null = final[col].notna().sum()
        unique = final[col].nunique(
            dropna=True
        )

        # USD/IDR sengaja dipertahankan walaupun nilainya
        # sama untuk semua ticker. Fitur ini disimpan di CSV
        # sebagai variabel makro; nanti bisa dikeluarkan dari X
        # saat K-Means karena tidak membedakan antar saham.
        if col.startswith("USD_IDR_"):
            keep_numeric.append(col)
            continue

        if (
            non_null >= 2
            and unique >= 2
        ):
            keep_numeric.append(col)

    remove_low_info = [
        c
        for c in numeric_columns
        if c not in keep_numeric
    ]

    if remove_low_info:

        print()
        print(
            "Numeric feature minim informasi dibuang:"
        )

        for c in remove_low_info:
            print(f"  - {c}")

        final = final.drop(
            columns=remove_low_info
        )

    # Refresh numeric list.
    numeric_columns = [
        c
        for c in final.columns
        if c not in id_columns
        and pd.api.types.is_numeric_dtype(
            final[c]
        )
    ]

    # --------------------------------------------------------
    # DROP ROW DENGAN NUMERIC TERLALU BANYAK NULL
    # --------------------------------------------------------

    if numeric_columns:

        completeness = (
            final[numeric_columns]
            .notna()
            .mean(axis=1)
        )

        before = len(final)

        final = final[
            completeness
            >= MIN_ROW_COMPLETENESS
        ].copy()

        removed = before - len(final)

        print()
        print(
            "Ticker dibuang karena numeric "
            f"completeness < "
            f"{MIN_ROW_COMPLETENESS * 100:.0f}%: "
            f"{removed:,}"
        )

    # --------------------------------------------------------
    # IMPUTE NUMERIC NULL DENGAN MEDIAN
    # --------------------------------------------------------

    numeric_columns = [
        c
        for c in final.columns
        if c not in id_columns
        and pd.api.types.is_numeric_dtype(
            final[c]
        )
    ]

    for col in numeric_columns:

        median_value = final[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        final[col] = (
            final[col]
            .fillna(median_value)
        )

    # --------------------------------------------------------
    # SORT
    #
    # Ticker ascending.
    # --------------------------------------------------------

    final = (
        final
        .sort_values(
            "Ticker",
            kind="stable"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # NO: CUMA SATU
    # --------------------------------------------------------

    final.insert(
        0,
        "No",
        range(
            1,
            len(final) + 1
        )
    )

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

    first_columns = [
        "No",
        "Ticker",
        "Sector",
        "Sharia",
    ]

    stock_columns = [
        c
        for c in final.columns
        if c.startswith("Stock_")
    ]

    financial_columns = [
        c
        for c in final.columns
        if c in FINANCIAL_FEATURES
    ]

    fx_columns = [
        c
        for c in final.columns
        if c.startswith("USD_IDR_")
    ]

    remaining = [
        c
        for c in final.columns
        if c not in (
            first_columns
            + stock_columns
            + financial_columns
            + fx_columns
        )
    ]

    ordered = (
        first_columns
        + stock_columns
        + financial_columns
        + fx_columns
        + remaining
    )

    ordered = [
        c
        for c in ordered
        if c in final.columns
    ]

    ordered = list(
        dict.fromkeys(ordered)
    )

    final = final[
        ordered
    ]

    # --------------------------------------------------------
    # VALIDATION USD/IDR
    # --------------------------------------------------------

    usd_columns = [
        c for c in final.columns
        if c.startswith("USD_IDR_")
    ]

    if not usd_columns:
        raise ValueError(
            "VALIDATION ERROR: kolom USD/IDR tidak ditemukan "
            "di dataset final."
        )

    print()
    print(
        "USD/IDR features retained:",
        ", ".join(usd_columns)
    )

    # --------------------------------------------------------
    # VALIDATION 1: No hanya satu
    # --------------------------------------------------------

    no_count = sum(
        1
        for c in final.columns
        if str(c).strip().lower() == "no"
    )

    if no_count != 1:
        raise ValueError(
            f"VALIDATION ERROR: jumlah kolom No = "
            f"{no_count}, harus 1."
        )

    # --------------------------------------------------------
    # VALIDATION 2: No 1..N
    # --------------------------------------------------------

    expected_no = pd.Series(
        range(1, len(final) + 1),
        dtype="int64"
    )

    if not final["No"].reset_index(
        drop=True
    ).equals(expected_no):

        raise ValueError(
            "VALIDATION ERROR: No bukan 1..N."
        )

    # --------------------------------------------------------
    # VALIDATION 3: ticker unique
    # --------------------------------------------------------

    if final["Ticker"].duplicated().any():

        raise ValueError(
            "VALIDATION ERROR: Ticker masih duplicate."
        )

    # --------------------------------------------------------
    # VALIDATION 4: numeric tidak NULL
    # --------------------------------------------------------

    numeric_final = [
        c
        for c in final.columns
        if c not in id_columns
        and c != "No"
        and pd.api.types.is_numeric_dtype(
            final[c]
        )
    ]

    if numeric_final:

        total_numeric_null = int(
            final[numeric_final]
            .isna()
            .sum()
            .sum()
        )

        if total_numeric_null > 0:

            raise ValueError(
                "VALIDATION ERROR: "
                f"masih ada {total_numeric_null:,} "
                "numeric NULL."
            )

    # --------------------------------------------------------
    # VALIDATION 5: jumlah row = jumlah ticker unique
    # --------------------------------------------------------

    if len(final) != final["Ticker"].nunique():

        raise ValueError(
            "VALIDATION ERROR: bukan 1 row per ticker."
        )

    return final


# ============================================================
# SAVE
# ============================================================

def save_output(final):

    final.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
        na_rep=""
    )

    print()
    print("==============================================")
    print("SELESAI - CLUSTERING READY")
    print("==============================================")

    print(
        f"Rows    : {len(final):,}"
    )

    print(
        f"Columns : {len(final.columns):,}"
    )

    print(
        f"Tickers : {final['Ticker'].nunique():,}"
    )

    print()
    print(
        f"Output:\n{OUTPUT}"
    )

    print()
    print("Kolom final:")

    for col in final.columns:
        print(f"  - {col}")

    print()
    print("5 baris pertama:")
    print(
        final.head(5).to_string(
            index=False
        )
    )

    print()
    print("5 baris terakhir:")
    print(
        final.tail(5).to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print("BUILD CLUSTERING DATASET")
    print("2023 - 2024")
    print("==============================================")
    print(
        f"Folder: {BASE_DIR}"
    )

    # 1. Financial 24 Excel.
    financial = load_financial()

    # 2. Stock daily.
    stock = load_stock()

    # 3. Ringkas stock menjadi 1 row per ticker.
    stock_features = build_stock_features(
        stock
    )

    # 4. USD/IDR.
    fx = load_fx()

    # 5. Ringkas USD/IDR menjadi 1 row.
    fx_features = build_fx_features(
        fx
    )

    # 6. Build final.
    final = build_final(
        financial=financial,
        stock=stock,
        stock_features=stock_features,
        fx_features=fx_features,
    )

    # 7. Save.
    save_output(
        final
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nProgram dihentikan."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("==============================================")
        print("ERROR")
        print("==============================================")
        print(exc)

        sys.exit(1)
