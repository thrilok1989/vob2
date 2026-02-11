#!/usr/bin/env python3
"""Generate PDF report of complete app analysis inventory using reportlab."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def build_pdf():
    output_path = '/home/user/vob2/Complete_App_Analysis_Inventory.pdf'
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('TitleCustom', parent=styles['Title'], fontSize=24,
                                  textColor=colors.HexColor('#193C78'), spaceAfter=6)
    subtitle_style = ParagraphStyle('SubtitleCustom', parent=styles['Normal'], fontSize=14,
                                     textColor=colors.HexColor('#505050'), alignment=TA_CENTER, spaceAfter=4)
    section_style = ParagraphStyle('SectionCustom', parent=styles['Heading1'], fontSize=14,
                                    textColor=colors.HexColor('#193C78'), spaceBefore=12, spaceAfter=6,
                                    borderWidth=1, borderColor=colors.HexColor('#193C78'), borderPadding=3)
    subsection_style = ParagraphStyle('SubSectionCustom', parent=styles['Heading2'], fontSize=11,
                                       textColor=colors.HexColor('#325AA0'), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('BodyCustom', parent=styles['Normal'], fontSize=8.5,
                                 textColor=colors.HexColor('#282828'))
    cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontSize=7.5,
                                 textColor=colors.HexColor('#282828'), leading=9)
    cell_hdr_style = ParagraphStyle('CellHdrStyle', parent=styles['Normal'], fontSize=7.5,
                                     textColor=colors.white, leading=9, alignment=TA_CENTER)

    HDR_BG = colors.HexColor('#193C78')
    ROW_ALT = colors.HexColor('#F0F5FF')

    def make_table(headers, data, col_widths=None):
        """Build a styled table."""
        avail = doc.width
        if col_widths is None:
            col_widths = [avail / len(headers)] * len(headers)
        else:
            # Scale col_widths proportionally to available width
            total = sum(col_widths)
            col_widths = [w / total * avail for w in col_widths]

        tdata = [[Paragraph(h, cell_hdr_style) for h in headers]]
        for row in data:
            tdata.append([Paragraph(str(c), cell_style) for c in row])

        t = Table(tdata, colWidths=col_widths, repeatRows=1)
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), HDR_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ]
        for i in range(1, len(tdata)):
            if i % 2 == 0:
                style_cmds.append(('BACKGROUND', (0, i), (-1, i), ROW_ALT))
        t.setStyle(TableStyle(style_cmds))
        return t

    elements = []
    sp = Spacer(1, 6*mm)
    sp_sm = Spacer(1, 3*mm)

    # ═══════════════════════ COVER PAGE ═══════════════════════
    elements.append(Spacer(1, 30*mm))
    elements.append(Paragraph('NIFTY TRADING & OPTIONS ANALYZER', title_style))
    elements.append(Spacer(1, 8*mm))
    elements.append(Paragraph('Complete Analysis Inventory & Feature Documentation', subtitle_style))
    elements.append(Spacer(1, 6*mm))
    info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=10,
                                 textColor=colors.HexColor('#787878'), alignment=TA_CENTER)
    elements.append(Paragraph('App: vob.py (5,240 lines)  |  86 Total Features  |  Streamlit + Dhan API + Supabase', info_style))
    elements.append(Paragraph('Including: Aether Flow PineScript v6 Indicator (10 modules)', info_style))
    elements.append(Spacer(1, 15*mm))
    elements.append(Paragraph('Generated: February 2026', info_style))
    elements.append(PageBreak())

    # ═══════════════════════ A. TECHNICAL ANALYSIS ═══════════════════════
    elements.append(Paragraph('A. TECHNICAL ANALYSIS MODULES (Python-based)', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Module / Class', 'What It Does', 'Key Outputs'],
        [
            ['1', 'PivotIndicator', 'Multi-TF pivot detection (3M, 5M, 10M, 15M, 1H)', 'Swing High/Low pivot levels, S/R lines on chart'],
            ['2', 'VolumeOrderBlocks', 'Volume-based Order Block detection', 'Bullish OB (teal), Bearish OB (purple), S/R levels'],
            ['3', 'TriplePOC', 'Point of Control across 3 periods (10, 25, 70)', 'POC1 (pink), POC2 (blue), POC3 (green) steplines, signals'],
            ['4', 'FutureSwing', 'Swing projection with volume delta analysis', 'Direction (Bull/Bear), projected target %, volume delta'],
            ['5', 'ReversalDetector', 'Intraday reversal detection (bullish + bearish)', 'Score (0-6), verdict (Strong Buy to Strong Sell), entry rules'],
            ['6', 'RSIVolatilitySuppression', 'RSI volatility suppression zone detection', 'Suppression zones on chart, breakout signals (▲/▼)'],
        ],
        [12, 55, 110, 100]
    ))
    elements.append(sp)

    # ═══════════════════════ B. OPTIONS CHAIN ═══════════════════════
    elements.append(Paragraph('B. OPTIONS CHAIN ANALYSIS', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Analysis', 'What It Computes', 'Signal Output'],
        [
            ['6', '18-Factor Bias Scoring', 'LTP, OI, ChgOI, Vol, Delta, Gamma, Theta, Ask, Bid, IV, DVP, Pressure, DeltaExp, GammaExp, Level, PCR, PCR Signal, AskBid', 'Per-strike bias classification'],
            ['7', 'Max Pain Calculation', 'Strike where option buyers lose most (CE+PE pain)', 'Max Pain level as support'],
            ['8', 'OI Wall Detection', 'Identifies max CE OI (resistance) and max PE OI (support)', 'OI Wall Resistance & Support levels'],
            ['9', 'Fresh OI Buildup', 'Spots fresh CE/PE buildup from change-in-OI', 'Fresh CE Buildup (resistance), Fresh PE Buildup (support)'],
            ['10', 'Option Chain Verdict', 'Aggregate scoring of all bias factors', 'Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish'],
            ['11', 'Operator Entry Detection', 'OI + ChgOI alignment check', 'Entry Bull, Entry Bear, No Entry'],
            ['12', 'Fake vs Real Move', 'Validates if price move is backed by OI', 'Scalp / Momentum classification'],
        ],
        [12, 55, 110, 100]
    ))
    elements.append(sp_sm)

    elements.append(Paragraph('18 Bias Factors Detail', subsection_style))
    elements.append(make_table(
        ['#', 'Factor', '#', 'Factor', '#', 'Factor'],
        [
            ['1', 'LTP Bias', '7', 'Theta Bias', '13', 'DeltaExp (Delta Exposure)'],
            ['2', 'OI Bias', '8', 'AskQty Bias', '14', 'GammaExp (Gamma Exposure)'],
            ['3', 'ChgOI Bias', '9', 'BidQty Bias', '15', 'Level Detection (S/R/Neutral)'],
            ['4', 'Volume Bias', '10', 'AskBid Bias', '16', 'PCR (Put-Call Ratio)'],
            ['5', 'Delta Bias', '11', 'IV Bias', '17', 'PCR Signal (Bull/Bear/Neutral)'],
            ['6', 'Gamma Bias', '12', 'DVP Bias (Delta-Vol-Price)', '18', 'PressureBias (Bid-Ask)'],
        ],
        [10, 42, 10, 52, 10, 52]
    ))
    elements.append(PageBreak())

    # ═══════════════════════ C. GREEKS & GEX ═══════════════════════
    elements.append(Paragraph('C. GREEKS & GAMMA EXPOSURE (GEX) ANALYSIS', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Analysis', 'What It Computes', 'Signal Output'],
        [
            ['13', 'Black-Scholes Greeks', 'Delta, Gamma, Vega, Theta, Rho per strike', 'Full Greeks table (expandable)'],
            ['14', 'Net GEX Calculation', 'Dealer gamma exposure per strike', 'Net GEX value, histogram chart'],
            ['15', 'Gamma Flip Level', 'Strike where dealer gamma crosses zero', 'Flip level (key pivot for price)'],
            ['16', 'GEX Magnet', 'Strike with highest positive gamma (price attraction)', 'Magnet strike level'],
            ['17', 'GEX Repeller', 'Strike with most negative gamma (price acceleration)', 'Repeller strike level'],
            ['18', 'GEX Market Regime', 'Positive GEX = Pin/Chop; Negative GEX = Trend/Breakout', 'Regime label + interpretation'],
            ['19', 'GEX Time Series', 'Historical GEX tracking for ATM +/- 2 strikes (200 entries)', '5-column time series charts'],
            ['20', 'GEX Change Alerts', 'Monitors GEX sign flips, >30% changes, gamma flip crossings', 'Telegram alerts'],
        ],
        [12, 55, 110, 100]
    ))
    elements.append(sp)

    # ═══════════════════════ D. PCR ═══════════════════════
    elements.append(Paragraph('D. PCR (PUT-CALL RATIO) ANALYSIS', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Analysis', 'What It Computes', 'Signal Output'],
        [
            ['21', 'PCR per Strike', 'Put/Call OI ratio for each strike', 'PCR value per strike'],
            ['22', 'PCR Time Series', 'Historical PCR for ATM +/- 2 strikes (200 entries)', '5-column time series charts'],
            ['23', 'PCR Signal', 'Thresholds: >1.2 = Bullish, <0.7 = Bearish', 'Signal badge per column'],
            ['24', 'PCR x GEX Confluence', 'Cross-analysis of PCR + GEX for combined signal', 'STRONG BULL/BEAR, BULL RANGE, BEAR TREND badges'],
        ],
        [12, 55, 110, 100]
    ))
    elements.append(PageBreak())

    # ═══════════════════════ E. S/R ═══════════════════════
    elements.append(Paragraph('E. SUPPORT & RESISTANCE AGGREGATION (HTF S/R Panel)', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'S/R Source', 'Support Level', 'Resistance Level'],
        [
            ['25', 'Max Pain', 'Max Pain strike', '--'],
            ['26', 'OI Wall', 'Max PE OI strike', 'Max CE OI strike'],
            ['27', 'Gamma Exposure', 'PE Gamma support', 'CE Gamma resistance'],
            ['28', 'Delta Exposure', 'PE Delta support', 'CE Delta resistance'],
            ['29', 'Fresh OI Buildup', 'Fresh PE buildup', 'Fresh CE buildup'],
            ['30', 'Market Depth', 'Max BidQty strike', 'Max AskQty strike'],
            ['31', 'Pivot Support (5M)', '5-min pivot lows', '5-min pivot highs'],
            ['32', 'Pivot Support (15M)', '15-min pivot lows', '15-min pivot highs'],
            ['33', 'Pivot Support (1H)', '1-hr pivot lows', '1-hr pivot highs'],
            ['34', 'VOB Zones', 'Volume OB support', 'Volume OB resistance'],
        ],
        [12, 55, 105, 105]
    ))
    elements.append(sp)

    # ═══════════════════════ F. CHARTS ═══════════════════════
    elements.append(Paragraph('F. CHART VISUALIZATIONS', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Chart Type', 'Library', 'Location'],
        [
            ['35', 'Candlestick + Volume (OHLCV with overlays)', 'Plotly', 'Left column (main)'],
            ['36', 'Pivot Level Overlays (multi-TF lines)', 'Plotly', 'On candlestick chart'],
            ['37', 'VWAP Line (yellow dotted)', 'Plotly', 'On candlestick chart'],
            ['38', 'VOB Zones (colored rectangles)', 'Plotly', 'On candlestick chart'],
            ['39', 'Triple POC Lines (3 horizontal levels)', 'Plotly', 'On candlestick chart'],
            ['40', 'Net GEX Histogram (strike-by-strike bars)', 'Plotly', 'Right column'],
            ['41', 'PCR Time Series (5 line charts)', 'Plotly', 'Right column'],
            ['42', 'GEX Time Series (5 line charts)', 'Plotly', 'Right column'],
            ['43', '30-Day Price Trend', 'Plotly', 'Analytics dashboard'],
            ['44', '30-Day Volume Trend', 'Plotly', 'Analytics dashboard'],
        ],
        [12, 80, 25, 160]
    ))
    elements.append(PageBreak())

    # ═══════════════════════ G. DASHBOARD ═══════════════════════
    elements.append(Paragraph('G. DASHBOARD & METRICS COMPONENTS', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Component', 'Data Shown'],
        [
            ['45', 'Spot Price Card', 'Current price, change (pts + %), color-coded green/red'],
            ['46', 'Day High/Low', 'Intraday range'],
            ['47', 'Volume Metric', 'Current volume'],
            ['48', 'POC Position Table', 'Above/Below/Inside for each POC period'],
            ['49', 'Swing Projection Table', 'Direction, target, delta, historical swing %'],
            ['50', 'Reversal Score Panel', 'Bullish score (0-6), Bearish score (0-6), individual signals'],
            ['51', 'Trading Psychology Panel', 'Entry guidelines and rules'],
            ['52', 'Option Chain Bias Table', 'All 18 bias factors per strike'],
            ['53', 'GEX Summary Cards', 'Net GEX, Flip, Magnet, Repeller'],
            ['54', 'GEX Breakdown Table', 'Call GEX, Put GEX, Net GEX per strike'],
            ['55', 'Greeks Table (expandable)', 'All greeks + IV + bid/ask per strike'],
            ['56', '30-Day Analytics Dashboard', 'Avg price, volatility, max gain/loss'],
        ],
        [12, 65, 200]
    ))
    elements.append(sp)

    # ═══════════════════════ H. ALERTS ═══════════════════════
    elements.append(Paragraph('H. ALERT & NOTIFICATION SYSTEM', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Alert Type', 'Trigger Condition', 'Channel'],
        [
            ['57', 'Trading Signal Alert', 'Bullish/Bearish pivot signal detected', 'Telegram'],
            ['58', 'ATM Verdict Alert', 'Strong Bullish/Bearish verdict triggered', 'Telegram'],
            ['59', 'GEX Sign Flip', 'Net GEX crosses zero (regime change)', 'Telegram'],
            ['60', 'Large GEX Change', 'Delta GEX exceeds 30% threshold', 'Telegram'],
            ['61', 'Gamma Flip Crossed', 'Price crosses gamma flip strike level', 'Telegram'],
        ],
        [12, 55, 120, 90]
    ))
    elements.append(sp)

    # ═══════════════════════ I. SIDEBAR ═══════════════════════
    elements.append(Paragraph('I. SIDEBAR CONFIGURATION', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Setting', 'Options'],
        [
            ['62', 'Timeframe selector', '1m, 3m, 5m, 10m, 15m'],
            ['63', 'Pivot toggles', '3M, 5M, 10M, 15M (individual on/off)'],
            ['64', 'Pivot proximity slider', '+/- 1-20 points'],
            ['65', 'Expiry date selector', 'Weekly/Monthly expiry list from Dhan API'],
            ['66', 'Historical days slider', '1-5 days of candle data'],
            ['67', 'Use Cached Data toggle', 'Cache (Supabase) vs Live API'],
            ['68', 'Auto-refresh toggle', '2-min interval option'],
            ['69', 'Telegram signal toggle', 'Enable/disable push notifications'],
            ['70', 'Database cleanup', 'Clear 7/14/30 day old data from Supabase'],
            ['71', 'Save Preferences', 'Persist user settings to Supabase'],
            ['72', 'Analytics Dashboard toggle', 'Show/hide 30-day analytics panel'],
        ],
        [12, 65, 200]
    ))
    elements.append(PageBreak())

    # ═══════════════════════ J. PINESCRIPT ═══════════════════════
    elements.append(Paragraph('J. AETHER FLOW PINESCRIPT v6 INDICATOR (10 Modules)', section_style))
    elements.append(Paragraph('External TradingView indicator: SMC Flow System v1.2 by Fatich.id', body_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Module', 'In-Chart Feature'],
        [
            ['73', 'Mxwll Suite', 'CHoCH / BOS / I-CHoCH / I-BOS + HH/HL/LH/LL swing labels'],
            ['74', 'Auto Fibonacci', '10 fib levels (0.236 to 1.618) auto-drawn from swing pivots'],
            ['75', 'Session Heatmap', 'NY (red) / Asia (green) / London (gold) background colors'],
            ['76', 'Session Dashboard', 'Real-time session timer + volume activity classification (5-tier)'],
            ['77', 'UT Bot Alerts', 'ATR trailing stop Buy/Sell signals with label management'],
            ['78', 'Hull Suite', 'HMA/EHMA/THMA trend bands (default length 55)'],
            ['79', 'LuxAlgo FVG', 'Fair Value Gap detection + mitigation tracking + dashboard'],
            ['80', 'LuxAlgo Order Blocks', 'Volume-pivot Order Block zones with wick/close mitigation'],
            ['81', 'Three-Bar Reversal', 'Pattern recognition with 4 trend filters (MA, ST, DC, None)'],
            ['82', 'Reversal Signals', 'TD-Sequential-style 9-count momentum phases with S/R levels'],
        ],
        [12, 55, 210]
    ))
    elements.append(sp)

    elements.append(Paragraph('PineScript Indicator - 12 Alert Conditions', subsection_style))
    elements.append(make_table(
        ['Alert Message', 'Trigger', 'Frequency'],
        [
            ['UT Long', 'UT Bot Buy signal fires', 'Once per bar'],
            ['UT Short', 'UT Bot Sell signal fires', 'Once per bar'],
            ['Hull trending up', 'MHULL crosses above SHULL', 'Once per bar'],
            ['Hull trending down', 'SHULL crosses above MHULL', 'Once per bar'],
            ['Bullish FVG detected', 'New bullish Fair Value Gap forms', 'Once per bar'],
            ['Bearish FVG detected', 'New bearish Fair Value Gap forms', 'Once per bar'],
            ['Bullish FVG mitigated', 'Bullish FVG mitigated by close', 'Once per bar'],
            ['Bearish FVG mitigated', 'Bearish FVG mitigated by close', 'Once per bar'],
            ['Bullish OB Formed', 'Bullish order block detected via volume pivot', 'Once per bar'],
            ['Bearish OB Formed', 'Bearish order block detected via volume pivot', 'Once per bar'],
            ['Bullish OB Mitigated', 'Bullish OB mitigated (wick or close)', 'Once per bar'],
            ['Bearish OB Mitigated', 'Bearish OB mitigated (wick or close)', 'Once per bar'],
        ],
        [55, 130, 92]
    ))
    elements.append(sp)

    elements.append(Paragraph('PineScript Indicator - Input Parameters (80 total)', subsection_style))
    elements.append(make_table(
        ['Group', 'Count', 'Key Parameters'],
        [
            ['Smart Money Concepts', '8', 'Bull/Bear color, Show Int/Ext, Sensitivity (3/5/8, 10/25/50), Structure type'],
            ['Swing Labels', '2', 'Show HH/LH, Show LH/LL'],
            ['High/Low', '4', 'Show 1D/4H levels and labels'],
            ['Auto Fibs', '31', '10 show toggles, 10 fib values (0.236-1.618), 10 colors, master toggle'],
            ['Sessions', '4', 'NY/Asia/London colors, Transparency (0-100)'],
            ['UT Bot Settings', '6', 'Key Value (2), ATR Period (6), Heikin Ashi mode, Label mgmt, Max labels (3)'],
            ['Hull Suite Settings', '10', 'Source, Variation (HMA/EHMA/THMA), Length (55), Multiplier, HTF, Colors, Band'],
            ['FVG', '12', 'Threshold %, Auto, Unmitigated levels, Mitigation, Timeframe, Extend, Dynamic, Colors, Dash'],
            ['LuxAlgo Order Blocks', '12', 'Vol Pivot Length (5), Bull/Bear counts (3), Colors, Line style/width, Mitigation'],
            ['Three Bar Reversal', '14', 'Pattern type, S/R mode, Colors, Trend filter type, MA type/lengths, ST factor, DC len'],
            ['Reversal Signals', '11', 'Momentum/Exhaustion display, S/R levels, Risk levels, Targets, Trade setup, Warnings'],
        ],
        [55, 15, 207]
    ))
    elements.append(PageBreak())

    # ═══════════════════════ K. DATA SOURCES ═══════════════════════
    elements.append(Paragraph('K. DATA SOURCES & APIs', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['#', 'Source', 'What It Provides', 'Integration Method'],
        [
            ['83', 'Dhan API (dhanhq)', 'Live OHLCV candles, LTP, option chain, greeks, expiry list', 'REST API with rate limiting + exponential backoff (2s,4s,8s,16s)'],
            ['84', 'Supabase', 'Persistent storage: candles, preferences, analytics history', 'supabase-py client library with auto-cleanup'],
            ['85', 'yfinance', 'Fallback/alternative price data source', 'yfinance library (pip package)'],
            ['86', 'Black-Scholes (scipy)', 'Greeks calculation engine using norm.cdf', 'scipy.stats.norm for analytical solutions'],
        ],
        [12, 50, 115, 100]
    ))
    elements.append(sp)

    # ═══════════════════════ SUMMARY ═══════════════════════
    elements.append(Paragraph('SUMMARY TOTALS', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['Category', 'Count'],
        [
            ['Technical Analysis Modules (Python)', '5'],
            ['Options Analysis Features', '7'],
            ['Greeks & GEX Analyses', '8'],
            ['PCR Analyses', '4'],
            ['S/R Level Sources', '10'],
            ['Charts / Visualizations', '10'],
            ['Dashboard Components', '12'],
            ['Alert Types (Telegram)', '5'],
            ['Sidebar Settings', '11'],
            ['PineScript Indicator Modules', '10'],
            ['PineScript Alert Conditions', '12'],
            ['Data Sources', '4'],
            ['TOTAL UNIQUE ANALYSES / FEATURES', '86'],
        ],
        [130, 50]
    ))
    elements.append(sp)

    # Verdict Scale
    elements.append(Paragraph('Verdict Scoring Scale', subsection_style))
    elements.append(make_table(
        ['Score Range', 'Verdict', 'Action'],
        [
            ['>= +4', 'Strong Bullish', 'High-confidence long entry'],
            ['>= +2', 'Bullish', 'Moderate long bias'],
            ['-1 to +1', 'Neutral', 'No directional bias'],
            ['<= -2', 'Bearish', 'Moderate short bias'],
            ['<= -4', 'Strong Bearish', 'High-confidence short entry'],
        ],
        [50, 60, 70]
    ))
    elements.append(sp)

    # Confluence Matrix
    elements.append(Paragraph('PCR x GEX Confluence Matrix', subsection_style))
    elements.append(make_table(
        ['PCR Signal', 'GEX Regime', 'Confluence', 'Meaning'],
        [
            ['Bullish (>1.2)', 'Positive (Pin/Chop)', 'BULL RANGE', 'Bullish bias, range-bound pinning expected'],
            ['Bullish (>1.2)', 'Negative (Trend)', 'STRONG BULL', 'Bullish with breakout potential'],
            ['Bearish (<0.7)', 'Positive (Pin/Chop)', 'BEAR RANGE', 'Bearish bias, range-bound movement'],
            ['Bearish (<0.7)', 'Negative (Trend)', 'STRONG BEAR', 'Bearish with breakdown potential'],
            ['Neutral (0.7-1.2)', 'Positive', 'NEUTRAL PIN', 'No direction, tight range expected'],
            ['Neutral (0.7-1.2)', 'Negative', 'VOLATILE', 'No direction, large swings possible'],
        ],
        [45, 50, 40, 100]
    ))
    elements.append(PageBreak())

    # ═══════════════════════ TECH STACK ═══════════════════════
    elements.append(Paragraph('TECHNOLOGY STACK & DEPENDENCIES', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['Package', 'Version', 'Purpose'],
        [
            ['streamlit', '>=1.28.0', 'Web application framework (main UI)'],
            ['streamlit-autorefresh', '>=0.1.6', 'Auto-refresh component (30-second interval)'],
            ['requests', '>=2.31.0', 'HTTP client for API calls'],
            ['pandas', '>=2.1.0', 'Data manipulation and analysis'],
            ['numpy', '>=1.24.0', 'Numerical computation'],
            ['scipy', '>=1.11.0', 'Black-Scholes Greeks calculation (norm.cdf)'],
            ['plotly', '>=5.17.0', 'Interactive charts and visualizations'],
            ['pytz', '>=2023.3', 'Timezone handling (Asia/Kolkata IST)'],
            ['openpyxl', '>=3.1.0', 'Excel file handling'],
            ['supabase', '>=1.0.3', 'Database client (Supabase persistence)'],
            ['dhanhq', '>=2.0.2', 'Dhan trading API client'],
            ['yfinance', '>=0.2.28', 'Yahoo Finance data (fallback source)'],
            ['pandas_ta', '>=0.3.14', 'Technical analysis indicators library'],
        ],
        [50, 35, 192]
    ))
    elements.append(sp)

    elements.append(Paragraph('MARKET HOURS & OPERATIONAL INFO', section_style))
    elements.append(sp_sm)
    elements.append(make_table(
        ['Parameter', 'Value'],
        [
            ['Trading Hours', '8:30 AM - 3:45 PM IST (Monday - Friday)'],
            ['Timezone', 'Asia/Kolkata (IST)'],
            ['Auto-Refresh Interval', '30 seconds (configurable)'],
            ['PCR History Depth', 'Last 200 entries per strike'],
            ['GEX History Depth', 'Last 200 entries per strike'],
            ['Historical Days (configurable)', '1-5 days of candle data'],
            ['Cached Data', 'Available outside market hours via Supabase'],
            ['API Rate Limiting', 'Exponential backoff: 2s, 4s, 8s, 16s on failure'],
        ],
        [70, 207]
    ))

    # Build
    doc.build(elements)
    return output_path


if __name__ == '__main__':
    path = build_pdf()
    print(f'PDF generated: {path}')
