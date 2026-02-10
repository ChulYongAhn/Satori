"""
S&P 500 전체 종목 리스트 (2024년 기준)
"""

SP500_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "CSCO", "ACN", "TXN", "QCOM", "INTC", "AMD", "INTU", "IBM",
    "NOW", "AMAT", "PANW", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "ADSK", "FTNT",
    "MCHP", "MU", "NXPI", "MRVL", "MSCI", "FICO", "ANSS", "EA", "PAYX", "ROP",
    "WDC", "STX", "HPQ", "DELL", "HPE", "NTAP", "TYL", "PTC", "ZBRA", "AKAM",
    "FSLR", "ENPH", "TER", "KEYS", "TRMB", "FFIV", "JNPR", "VRSN", "QRVO", "SWKS",

    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "CVS", "GILD", "MDT", "ELV", "CI", "SYK", "ZTS", "ISRG", "VRTX",
    "REGN", "BSX", "HUM", "EW", "MCK", "HCA", "BDX", "MRNA", "BIIB", "CNC",
    "IDXX", "DXCM", "ILMN", "MTD", "IQV", "A", "RMD", "WST", "HOLX", "BAX",
    "ABC", "ZBH", "ALGN", "WAT", "LH", "DGX", "PKI", "TECH", "CTLT", "CRL",
    "COO", "MOH", "VTRS", "HSIC", "DVA", "UHS", "XRAY", "OGN", "BIO", "INCY",

    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "SPGI", "GS", "MS", "BLK",
    "C", "AXP", "SCHW", "CB", "PGR", "CME", "USB", "PNC", "MMC", "TFC",
    "ICE", "AON", "COF", "AJG", "MCO", "TRV", "AFL", "ALL", "AIG", "MET",
    "PRU", "MSCI", "FIS", "DFS", "STT", "BK", "TROW", "FITB", "SYF", "SIVB",
    "MTB", "RJF", "NTRS", "CFG", "HBAN", "RF", "KEY", "WRB", "CINF", "FRC",
    "L", "PFG", "ZION", "CBOE", "NDAQ", "RE", "MKTX", "GL", "AIZ", "SBNY",

    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
    "ORLY", "AZO", "MAR", "GM", "HLT", "RCL", "F", "DHI", "LEN", "YUM",
    "ROST", "DG", "DLTR", "GRMN", "DPZ", "ULTA", "LVS", "MGM", "CCL", "NCLH",
    "EXPE", "EBAY", "ETSY", "APTV", "BWA", "POOL", "KMX", "BBY", "LKQ", "WHR",
    "PHM", "NVR", "MHK", "RL", "TPR", "HAS", "WYNN", "CZR", "PENN", "AAP",

    # Consumer Staples
    "PG", "WMT", "KO", "PEP", "COST", "PM", "MDLZ", "MO", "CL", "EL",
    "KMB", "GIS", "ADM", "SJM", "K", "MKC", "HSY", "CHD", "KDP", "STZ",
    "MNST", "KR", "SYY", "TSN", "HRL", "CPB", "CAG", "BF.B", "CLX", "WBA",
    "TAP", "LW", "BG", "DGX", "COTY",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PXD", "PSX", "VLO", "WMB",
    "OKE", "HES", "KMI", "FANG", "DVN", "TRGP", "HAL", "BKR", "OXY", "APA",
    "CTRA", "MRO", "EQT", "SW", "HFC",

    # Industrials
    "UPS", "RTX", "CAT", "BA", "HON", "UNP", "LMT", "DE", "GE", "MMM",
    "WM", "EMR", "ETN", "ITW", "CSX", "NSC", "FDX", "PH", "JCI", "GD",
    "TT", "CARR", "OTIS", "CMI", "PCAR", "ROK", "LHX", "AME", "DOV", "VRSK",
    "FTV", "RSG", "ODFL", "FAST", "SWK", "URI", "CPRT", "EFX", "EXPD", "XYL",
    "IR", "AOS", "WAB", "GNRC", "IEX", "TXT", "J", "MAS", "SNA", "CHRW",
    "JBHT", "PWR", "HII", "LDOS", "TDG", "NDSN", "RHI", "ALLE", "DAL", "UAL",
    "LUV", "ALK", "AAL",

    # Materials
    "LIN", "APD", "SHW", "FCX", "ECL", "DD", "NEM", "DOW", "PPG", "NUE",
    "CTVA", "VMC", "MLM", "BALL", "AVY", "IP", "CF", "AMCR", "ALB", "EMN",
    "LYB", "FMC", "MOS", "SEE", "CE",

    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "SPG", "VICI",
    "AVB", "EQR", "SBAC", "WY", "ARE", "VTR", "INVH", "ESS", "MAA", "PEAK",
    "DRE", "BXP", "HST", "UDR", "CPT", "FRT", "REG", "AIV", "KIM",

    # Utilities
    "NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "WEC",
    "PEG", "ES", "PCG", "EIX", "FE", "PPL", "AEE", "ETR", "AWK", "DTE",
    "CMS", "CNP", "ATO", "AES", "NI", "LNT", "EVRG", "PNW", "NRG",

    # Communication Services
    "GOOGL", "GOOG", "META", "DIS", "CMCSA", "NFLX", "VZ", "T", "TMUS", "CHTR",
    "EA", "TTWO", "ATVI", "WBD", "PARA", "FOXA", "FOX", "MTCH", "NWSA", "NWS",
    "OMC", "IPG", "DISH"
]

# 거래소별 분류
NASDAQ_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "ADBE", "CSCO", "QCOM", "INTC", "AMD", "INTU", "TXN", "NFLX", "PYPL", "SBUX",
    "COST", "PEP", "CMCSA", "TMUS", "CHTR", "AMGN", "GILD", "ISRG", "VRTX", "REGN",
    "MRNA", "BIIB", "DXCM", "ILMN", "IDXX", "ALGN", "WBA", "ATVI", "EA", "MAR",
    "PAYX", "KLAC", "LRCX", "ASML", "SNPS", "CDNS", "MRVL", "PANW", "ADSK", "FTNT",
    "WDAY", "TEAM", "CRWD", "ZS", "OKTA", "DOCU", "ROST", "DLTR", "ORLY", "KDP",
    "MNST", "FAST", "VRSK", "CPRT", "ODFL", "PCAR", "CTAS", "MCHP", "MU", "KHC",
    "MDLZ", "ADP", "ADI", "BKNG", "NXPI", "LULU", "EXC", "XEL", "AEP", "WBD",
    "FOXA", "FOX", "NWSA", "NWS", "PARA", "DISH", "EBAY", "SIRI", "LCID", "RIVN"
}


def get_exchange(ticker: str) -> str:
    """
    티커의 거래소 반환

    Parameters:
    -----------
    ticker: 종목 티커

    Returns:
    --------
    "NAS" (NASDAQ) 또는 "NYS" (NYSE)
    """
    return "NAS" if ticker in NASDAQ_TICKERS else "NYS"


def get_sp500_full_list() -> list:
    """S&P 500 전체 종목 리스트 반환"""
    return SP500_TICKERS.copy()


def get_sp500_by_sector(sector: str) -> list:
    """
    섹터별 S&P 500 종목 리스트 반환

    Sectors: Technology, Healthcare, Financials, Consumer Discretionary,
             Consumer Staples, Energy, Industrials, Materials, Real Estate,
             Utilities, Communication Services
    """
    # 섹터별 분류는 위 리스트의 주석을 참고하여 구현
    # 실제 구현시 더 정확한 분류 필요
    pass


if __name__ == "__main__":
    print(f"Total S&P 500 tickers: {len(SP500_TICKERS)}")
    print(f"NASDAQ tickers: {len(NASDAQ_TICKERS)}")
    print(f"NYSE tickers: {len(SP500_TICKERS) - len([t for t in SP500_TICKERS if t in NASDAQ_TICKERS])}")