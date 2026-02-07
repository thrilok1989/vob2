# AETHER FLOW SYSTEM – Mxwll Fusion Engine (LuxAlgo OB Edition)

**Version:** 1.2 (LuxAlgo OB + UT Label Management)
**Author:** Fatich.id
**PineScript Version:** v6
**Overlay:** Yes
**Limits:** max_labels=500, max_lines=500, max_boxes=500, max_bars_back=500

---

## 1. INDICATOR MODULES OVERVIEW

| # | Module Name | Purpose | Toggle Input | Default |
|---|-------------|---------|-------------|---------|
| 1 | Mxwll Suite (Smart Money Concepts) | Market structure detection – CHoCH/BOS, swing labels, HH/HL/LH/LL | `showInt`, `showExt` | ON, ON |
| 2 | Auto Fibonacci Levels | Auto-drawn fib retracement/extension from swing pivots | `showFibs` | ON |
| 3 | High/Low Levels (1D & 4H) | Previous Day and Rolling 4-Hour High/Low lines | `show1D`, `show4H` | ON, ON |
| 4 | Session Heatmap | Background coloring for NY / Asia / London sessions | via `tra` (transparency) | ON (98% transparent) |
| 5 | Session Dashboard Table | Real-time session info, countdown timers, volume activity | Always shown on `barstate.islast` | ON |
| 6 | UT Bot Alerts | ATR-based trailing stop trend-shift system with label management | `showUTLabels` | ON |
| 7 | Hull Suite | Hull Moving Average band for trend velocity/momentum | `visualSwitch` | ON |
| 8 | LuxAlgo FVG Engine | Fair Value Gap detection with mitigation tracking | `showLuxFVG` | ON |
| 9 | LuxAlgo Order Blocks | Volume-pivot based OB detection with mitigation | `showLuxOB` | ON |
| 10 | Three-Bar Reversal | Pattern recognition for 3-bar reversals with trend filters | `show3BR` | ON |
| 11 | Reversal Signals | Momentum phase counting (TD-Sequential style) with S/R levels | `showRS` | ON |

---

## 2. INPUT PARAMETERS – FULL TABULATION

### 2.1 Smart Money Concepts (Mxwll Suite)

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Bull Color | `bullC` | color | #14D990 (green) | Any color | Smart Money Concepts |
| Bear Color | `bearC` | color | #F24968 (red) | Any color | Smart Money Concepts |
| Show Internals | `showInt` | bool | true | true/false | Smart Money Concepts |
| Internals Sensitivity | `intSens` | int | 5 | 3, 5, 8 | Smart Money Concepts |
| Internal Structure | `intStru` | string | "All" | All, BoS, CHoCH | Smart Money Concepts |
| Show Externals | `showExt` | bool | true | true/false | Smart Money Concepts |
| Externals Sensitivity | `extSens` | int | 25 | 10, 25, 50 | Smart Money Concepts |
| External Structure | `extStru` | string | "All" | All, BoS, CHoCH | Smart Money Concepts |

### 2.2 Swing Labels

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Show HH/LH | `showHHLH` | bool | true | true/false | Swing Labels |
| Show LH/LL | `showHLLL` | bool | true | true/false | Swing Labels |

### 2.3 High/Low Levels

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Show Previous Day High | `show1D` | bool | true | true/false | High/Low |
| Show 1 Day Labels | `show1DLab` | bool | true | true/false | High/Low |
| Show 4 Hour High | `show4H` | bool | true | true/false | High/Low |
| Show 4 Hour Labels | `show4hLab` | bool | true | true/false | High/Low |

### 2.4 Auto Fibonacci Levels

| Parameter | Variable | Type | Default | Group |
|-----------|----------|------|---------|-------|
| Show Auto Fibs | `showFibs` | bool | true | Auto Fibs |
| Fib Level 1 | `fib1` | float | 0.236 | Auto Fibs |
| Fib Level 2 | `fib2` | float | 0.382 | Auto Fibs |
| Fib Level 3 | `fib3` | float | 0.500 | Auto Fibs |
| Fib Level 4 | `fib4` | float | 0.618 | Auto Fibs |
| Fib Level 5 | `fib5` | float | 0.786 | Auto Fibs |
| Fib Level 6 | `fib6` | float | 0.886 | Auto Fibs |
| Fib Level 7 | `fib7` | float | 1.130 | Auto Fibs |
| Fib Level 8 | `fib8` | float | 1.270 | Auto Fibs |
| Fib Level 9 | `fib9` | float | 1.410 | Auto Fibs |
| Fib Level 10 | `fib10` | float | 1.618 | Auto Fibs |
| Show toggles | `show236`..`show161` | bool | all true | Auto Fibs |
| Fib Colors | `fib1col`..`fib10col` | color | various | Auto Fibs |

**Fib Color Defaults:**

| Level | Color |
|-------|-------|
| 0.236 | rgb(160,165,185) (gray) |
| 0.382 | lime |
| 0.500 | yellow |
| 0.618 | orange |
| 0.786 | red |
| 0.886 | purple |
| 1.130 | blue |
| 1.270 | teal |
| 1.410 | maroon |
| 1.618 | navy |

### 2.5 Sessions

| Parameter | Variable | Type | Default | Group |
|-----------|----------|------|---------|-------|
| NY Color | `nyCol` | color | #f24968 (red) | Sessions |
| Asia Color | `asCol` | color | #14D990 (green) | Sessions |
| London Color | `loCol` | color | #F2B807 (gold) | Sessions |
| Transparency | `tra` | int | 98 | Sessions |

### 2.6 UT Bot Settings

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Key Value | `a` | int | 2 | Any | UT Bot Settings |
| ATR Period | `c` | int | 6 | Any | UT Bot Settings |
| Signals from Heikin Ashi | `h` | bool | false | true/false | UT Bot Settings |
| Show UT Labels | `showUTLabels` | bool | true | true/false | UT Bot Settings |
| Keep Recent Signals Only | `keepRecentOnly` | bool | true | true/false | UT Bot Settings |
| Max Recent Signals | `maxUTLabels` | int | 3 | 1–10 | UT Bot Settings |

### 2.7 Hull Suite Settings

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Hull Source | `srcHull` | source | close | Any source | Hull Suite Settings |
| Hull Variation | `modeSwitch` | string | "Hma" | Hma, Thma, Ehma | Hull Suite Settings |
| Hull Length | `length` | int | 55 | Any | Hull Suite Settings |
| Length Multiplier | `lengthMult` | float | 1.0 | Any | Hull Suite Settings |
| Show from HTF | `useHtf` | bool | false | true/false | Hull Suite Settings |
| Higher Timeframe | `htf` | timeframe | "240" | Any TF | Hull Suite Settings |
| Color by Trend | `switchColor` | bool | true | true/false | Hull Suite Settings |
| Color Candles | `candleCol` | bool | false | true/false | Hull Suite Settings |
| Show as Band | `visualSwitch` | bool | true | true/false | Hull Suite Settings |
| Line Thickness | `thicknesSwitch` | int | 1 | Any | Hull Suite Settings |
| Band Transparency | `transpSwitch` | int | 40 | Any | Hull Suite Settings |

### 2.8 FVG (Fair Value Gap)

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Show FVG | `showLuxFVG` | bool | true | true/false | FVG |
| Threshold % | `thresholdPer` | float | 0.0 | 0–100 | LuxAlgo FVG |
| Auto Threshold | `auto` | bool | false | true/false | FVG |
| Unmitigated Levels | `showLastFVG` | int | 0 | 0+ | FVG |
| Mitigation Levels | `mitigationLevels` | bool | false | true/false | FVG |
| Timeframe | `tfFVG` | timeframe | "" (current) | Any TF | FVG |
| Extend Bars | `extendFVG` | int | 20 | 0+ | FVG |
| Dynamic FVG | `dynamicFVG` | bool | false | true/false | FVG |
| Bullish FVG Color | `bullCss` | color | #089981 (70% transp) | Any | FVG |
| Bearish FVG Color | `bearCss` | color | #f23645 (70% transp) | Any | FVG |
| Show Dashboard | `showDash` | bool | false | true/false | FVG |
| Dashboard Location | `dashLoc` | string | "Top Right" | Top Right, Bottom Right, Bottom Left | LuxAlgo FVG |
| Text Size | `textSize` | string | "Small" | Tiny, Small, Normal | FVG |

### 2.9 LuxAlgo Order Blocks

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Show Order Blocks | `showLuxOB` | bool | true | true/false | LuxAlgo Order Blocks |
| Volume Pivot Length | `lengthOB` | int | 5 | 1+ | LuxAlgo Order Blocks |
| Bullish OB Count | `bull_ext_last` | int | 3 | 1+ | LuxAlgo Order Blocks |
| Bullish BG Color | `bg_bull_css` | color | #089981 (80% transp) | Any | LuxAlgo Order Blocks |
| Bullish Border Color | `bull_css` | color | #089981 | Any | LuxAlgo Order Blocks |
| Bullish Avg Line Color | `bull_avg_css` | color | #9598a1 (37% transp) | Any | LuxAlgo Order Blocks |
| Bearish OB Count | `bear_ext_last` | int | 3 | 1+ | LuxAlgo Order Blocks |
| Bearish BG Color | `bg_bear_css` | color | #f23645 (80% transp) | Any | LuxAlgo Order Blocks |
| Bearish Border Color | `bear_css` | color | #f23645 | Any | LuxAlgo Order Blocks |
| Bearish Avg Line Color | `bear_avg_css` | color | #9598a1 (37% transp) | Any | LuxAlgo Order Blocks |
| Average Line Style | `line_style` | string | "⎯⎯⎯" (solid) | ⎯⎯⎯, ----, ···· | LuxAlgo Order Blocks |
| Average Line Width | `line_width` | int | 1 | 1+ | LuxAlgo Order Blocks |
| Mitigation Method | `mitigationOB` | string | "Wick" | Wick, Close | LuxAlgo Order Blocks |

### 2.10 Three-Bar Reversal

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Show Three Bar Reversal | `show3BR` | bool | true | true/false | Three Bar Reversal |
| Pattern Type | `brpType` | string | "All" | Normal, Enhanced, All | Three Bar Reversal |
| Support/Resistance | `brpSR` | string | "Level" | Level, Zone, None | Three Bar Reversal |
| Bullish Pattern Color | `brpAC` | color | #2962ff (blue) | Any | Three Bar Reversal |
| Bearish Pattern Color | `brpSC` | color | #ff9800 (orange) | Any | Three Bar Reversal |
| Trend Filtering | `trendType3BR` | string | "None" | MA Cloud, Supertrend, Donchian, None | Three Bar Reversal |
| Trend Filter Mode | `trendFilt3BR` | string | "Aligned" | Aligned, Opposite | Three Bar Reversal |
| Bullish Trend Color | `trendAC3BR` | color | #089981 | Any | Three Bar Reversal |
| Bearish Trend Color | `trendSC3BR` | color | #f23645 | Any | Three Bar Reversal |
| MA Type | `maType3BR` | string | "HMA" | SMA, EMA, HMA, RMA, WMA, VWMA | Three Bar Reversal |
| Fast MA Length | `maFLength3BR` | int | 50 | 1–100 | Three Bar Reversal |
| Slow MA Length | `maSLength3BR` | int | 200 | 100+ | Three Bar Reversal |
| ST ATR Length | `atrPeriod3BR` | int | 10 | 1+ | Three Bar Reversal |
| ST Factor | `factor3BR` | float | 3.0 | 2+ (step 0.1) | Three Bar Reversal |
| DC Length | `length3BR` | int | 13 | 1+ | Three Bar Reversal |

### 2.11 Reversal Signals

| Parameter | Variable | Type | Default | Options/Range | Group |
|-----------|----------|------|---------|---------------|-------|
| Show Reversal Signals | `showRS` | bool | true | true/false | Reversal Signals |
| Display Momentum Phases | `bShRS` | string | "Completed" | Completed, Detailed, None | Reversal Signals |
| Support & Resistance Levels | `srLRS` | bool | true | true/false | Reversal Signals |
| Phase Display Style | `ptLTRS` | string | "Step Line w/ Diamonds" | Circles, Step Line, Step Line w/ Diamonds | Reversal Signals |
| Momentum Phase Risk Levels | `rsBRS` | bool | false | true/false | Reversal Signals |
| Risk Display Style | `ptSRRS` | string | "Circles" | Circles, Step Line | Reversal Signals |
| Display Exhaustion Phases | `eShRS` | string | "Completed" | Completed, Detailed, None | Reversal Signals |
| Exhaustion Phase Risk Levels | `rsERS` | bool | false | true/false | Reversal Signals |
| Exhaustion Phase Target Levels | `ttERS` | bool | false | true/false | Reversal Signals |
| Trade Setup Options | `tsoRS` | string | "None" | Momentum, Exhaustion, Qualified, None | Reversal Signals |
| Price Flip Warnings | `warRS` | bool | false | true/false | Reversal Signals |

---

## 3. CUSTOM TYPE DEFINITIONS

| Type Name | Fields | Purpose |
|-----------|--------|---------|
| `fvg` | `max` (float), `min` (float), `isbull` (bool), `t` (int=time) | Stores Fair Value Gap data |
| `barRS` | `o`, `h`, `l`, `c` (float), `i` (int=bar_index) | Bar OHLC data for Reversal Signals |
| `trbRS` | `bSC`, `bSH`, `bSL`, `sSC`, `sSH`, `sSL` | Trend/reversal bar state for RS |
| `treRS` | `bCC`, `bC8`, `bCHt`, `bCH`, `bCL`, `bCLt`, `bCD`, `sCC`, `sC8`, `sCHt`, `sCH`, `sCL`, `sCLt`, `sCT` | Exhaustion phase tracking for RS |
| `rollingTF` | `highTF`, `lowTF`, `highTFt`, `lowTFt`, `volTF`, `rTFdraw` (map), `rTFlabel` (map) | Rolling timeframe high/low/volume tracking |

---

## 4. CORE FUNCTIONS

| Function | Signature | Purpose |
|----------|-----------|---------|
| `tosolid()` | `method tosolid(color id) => color` | Converts a transparent color to fully opaque RGB |
| `movingAverage3BR()` | `(source, length, maType) => float` | Multi-type MA calculator (SMA/EMA/HMA/RMA/WMA/VWMA) |
| `isBullishReversal3BR()` | `() => bool` | Detects bullish 3-bar reversal pattern |
| `isBearishReversal3BR()` | `() => bool` | Detects bearish 3-bar reversal pattern |
| `f_xLXRS()` | `(_p, _l) => bool` | Cross-level check for Reversal Signals |
| `f_lnSRS()` | `(_s) => plot_style` | Maps string to plot style for RS display |
| `detectFVG()` | `() => [bool, bool, fvg]` | Detects bullish/bearish Fair Value Gaps with threshold |
| `get_line_style()` | `(style) => line_style` | Converts string to Pine line style constant |
| `get_coordinates()` | `(condition, top, btm, ob_val) => [arrays..., float]` | Captures OB coordinates on volume pivot detection |
| `remove_mitigated()` | `(ob_top, ob_btm, ob_left, ob_avg, target, bull) => bool` | Removes mitigated order blocks from arrays |
| `set_order_blocks()` | `(ob_top, ob_btm, ob_left, ob_avg, ext_last, bg, border, lvl) => void` | Renders OB boxes and average lines |
| `calculatePivots()` | `(lengthCalc) => [float, float]` | Calculates swing high/low pivots for structure |
| `drawChar()` | `(x, y, str, col, down) => void` | Draws structure labels (CHoCH/BOS) with dashed lines |
| `drawStructureExt()` | `() => int` | Draws external market structure (HH/HL/LH/LL + CHoCH/BOS) |
| `updateMain()` | `method (line id) => int` | Updates the main fib reference line between swing points |
| `quickLine()` | `(x2, y, color) => line` | Helper to draw a horizontal fib level line |
| `quickLabel()` | `(y, txt, color) => label` | Helper to draw a fib level label |
| `drawFibs()` | `() => void` | Renders all 10 fibonacci levels on last bar |
| `drawStructureInternals()` | `() => void` | Draws internal structure (I-CHoCH / I-BOS) |
| `tfDraw()` | `method (tfDiff, showLab, tf, showLevels) => [float, array]` | Rolling timeframe high/low drawing (higher TF bars) |
| `tfDrawLower()` | `(showLab, tf, showLevels) => [float, array]` | Rolling TF high/low using `request.security_lower_tf` |
| `calculateTimeDifference()` | `(ts1, ts2) => [hours, minutes]` | Computes hours/minutes between two timestamps |
| `timeIsInRange()` | `(startH, startM, endH, endM) => bool` | Checks if current NY time is within a range (realtime) |
| `timeIsInRange2()` | `(startH, startM, endH, endM) => bool` | Checks if bar time is within a range (historical) |
| `getActivity()` | `method (array, value) => string` | Classifies volume into Very Low/Low/Average/High/Very High |
| `HMA()` | `(src, length) => float` | Hull Moving Average calculation |
| `EHMA()` | `(src, length) => float` | Ehlers Hull Moving Average |
| `THMA()` | `(src, length) => float` | Triple Hull Moving Average |
| `Mode()` | `(mode, src, len) => float` | Hull variation selector |

---

## 5. ALERT CONDITIONS

| Alert Message | Trigger Condition | Frequency |
|--------------|-------------------|-----------|
| "UT Long" | UT Bot Buy signal fires | Once per bar |
| "UT Short" | UT Bot Sell signal fires | Once per bar |
| "Hull trending up." | MHULL crosses above SHULL | Once per bar |
| "Hull trending down." | SHULL crosses above MHULL | Once per bar |
| "Bullish FVG detected" | New bullish FVG forms | Once per bar |
| "Bearish FVG detected" | New bearish FVG forms | Once per bar |
| "Bullish FVG mitigated" | Bullish FVG gets mitigated | Once per bar |
| "Bearish FVG mitigated" | Bearish FVG gets mitigated | Once per bar |
| "Bullish OB Formed" | Bullish order block detected | Once per bar |
| "Bearish OB Formed" | Bearish order block detected | Once per bar |
| "Bullish OB Mitigated" | Bullish OB gets mitigated | Once per bar |
| "Bearish OB Mitigated" | Bearish OB gets mitigated | Once per bar |

---

## 6. VISUAL OUTPUTS (PLOTS & DRAWINGS)

### 6.1 Plot Outputs

| Plot | Variable/Expression | Color | Style | Display |
|------|---------------------|-------|-------|---------|
| MHULL | `MHULL` | Green (#00ff00) or Red (#ff0000) by trend | Line | Visible |
| SHULL | `SHULL` (if band mode) | Same as MHULL | Line | Conditional |
| Hull Band Fill | Between Fi1 and Fi2 | hullColor @ `transpSwitch` | Fill | Conditional |
| Bull FVG Max | `max_bull_fvg` | na (hidden) | — | display.none |
| Bull FVG Min | `min_bull_fvg` | na (hidden) | — | display.none |
| Bull FVG Fill | Between max/min bull plots | `bullCss` | Fill | Visible (dynamic mode) |
| Bear FVG Max | `max_bear_fvg` | na (hidden) | — | display.none |
| Bear FVG Min | `min_bear_fvg` | na (hidden) | — | display.none |
| Bear FVG Fill | Between max/min bear plots | `bearCss` | Fill | Visible (dynamic mode) |
| Bull OB | `bull_ob_plot` | `bull_css` | linebr, width=2 | display.none |
| Bear OB | `bear_ob_plot` | `bear_css` | linebr, width=2 | display.none |
| Bullish Momentum | `showBullMomentum` | #089981 (25% transp) | labelup | Below bar |
| Bearish Momentum | `showBearMomentum` | #f23645 (25% transp) | labeldown | Above bar |
| RS Resistance | `bSRRS` | #f23645 (50% transp) | Line, width=2 | Visible |
| RS Support | `sSSRS` | #089981 (50% transp) | Line, width=2 | Visible |
| UT Buy (no label mgmt) | `buy` signal | Green | labelup | Below bar |
| UT Sell (no label mgmt) | `sell` signal | Red | labeldown | Above bar |

### 6.2 Background Colors

| Condition | Color | Transparency |
|-----------|-------|--------------|
| NY Session (9:30–16:00 ET) | `nyCol` (#f24968) | `tra` (98) |
| Asia Session (20:00–02:00 ET) | `asCol` (#14D990) | `tra` (98) |
| London Session (03:00–11:30 ET) | `loCol` (#F2B807) | `tra` (98) |

### 6.3 Bar Coloring

| Condition | Color |
|-----------|-------|
| UT Bot bullish (price > trailing stop) | Green |
| UT Bot bearish (price < trailing stop) | Red |
| Hull candle coloring (if enabled) | hullColor (green/red by trend) |

### 6.4 Dynamic Drawing Objects

| Object Type | Module | Description |
|-------------|--------|-------------|
| Labels | Mxwll Suite | HH, HL, LH, LL swing labels |
| Labels + Lines | Mxwll Suite | CHoCH / BOS / I-CHoCH / I-BOS structure labels with dashed lines |
| Lines | High/Low | 1D and 4H rolling high/low horizontal lines (aqua) |
| Labels | High/Low | "240H", "240L", "1DH", "1DL" text labels |
| Line (dashed) | Auto Fibs | Main swing reference line (green=bullish, red=bearish) |
| Lines + Labels | Auto Fibs | 10 fibonacci level lines with value labels |
| Labels | UT Bot | Buy/Sell labels (managed array, max=`maxUTLabels`) |
| Boxes | FVG | Fair Value Gap zone boxes (bullish/bearish) |
| Lines | FVG | Mitigation level dashed lines |
| Lines | FVG | Unmitigated level lines |
| Boxes | Order Blocks | OB zone boxes (bull=green bg, bear=red bg) |
| Lines | Order Blocks | OB average price lines (configurable style) |
| Labels + Lines + Boxes + Linefills | Three-Bar Reversal | Pattern markers (▲/▼), zone lines, S/R boxes, confirmation dots |
| Table | Session Dashboard | 2-column table with session info, timers, volume activity |
| Table | FVG Dashboard | 3x3 table with FVG counts and mitigation percentages |

---

## 7. SESSION SCHEDULE (New York Time)

| Session | Start (ET) | End (ET) | BG Color | Dashboard Label |
|---------|-----------|---------|----------|-----------------|
| New York | 09:30 | 16:00 | #f24968 (red) | "New York" |
| Asia | 20:00 | 02:00 (+1d) | #14D990 (green) | "Asia" |
| London | 03:00 | 11:30 | #F2B807 (gold) | "London" |
| Dead Zone | All other times | — | None | "Dead Zone" |

---

## 8. SESSION DASHBOARD TABLE (top_right)

| Row | Column 0 (Label) | Column 1 (Value) |
|-----|-------------------|-------------------|
| 2 | Session: | Current session name |
| 3 | Session Close: | Hours + Minutes countdown (or "Dead Zone") |
| 4 | Next Session: | Next session name |
| 5 | Next Session Open: | Hours + Minutes countdown |
| 6 | 4-Hr Volume: | Activity level (Very Low / Low / Average / High / Very High) |
| 7 | 24-Hr Volume: | Activity level (Very Low / Low / Average / High / Very High) |

---

## 9. VOLUME ACTIVITY CLASSIFICATION

| Percentile Range | Activity Label |
|-----------------|----------------|
| ≤ 10th percentile | Very Low |
| ≤ 33rd percentile | Low |
| ≤ 50th percentile | Average |
| ≤ 66th percentile | High |
| > 66th percentile | Very High |

---

## 10. MARKET STRUCTURE LOGIC

### External Structure (Mxwll Suite)

| Event | Detection | Label | Visual |
|-------|-----------|-------|--------|
| Higher High | New swing high > previous swing high | "HH" (bearC, above) | Label |
| Lower High | New swing high < previous swing high | "LH" (bearC, above) | Label |
| Higher Low | New swing low > previous swing low | "HL" (bullC, below) | Label |
| Lower Low | New swing low < previous swing low | "LL" (bullC, below) | Label |
| Break of Structure (BOS) | Price crosses swing level in same trend direction | "BoS" + dashed line | Label + Line |
| Change of Character (CHoCH) | Price crosses swing level against trend | "CHoCH" + dashed line | Label + Line |

### Internal Structure

| Event | Label | Visual |
|-------|-------|--------|
| Internal BOS | "I-BoS" | Label + dashed line |
| Internal CHoCH | "I-CHoCH" | Label + dashed line |

---

## 11. UT BOT SYSTEM LOGIC

| Component | Formula/Logic |
|-----------|--------------|
| ATR | `ta.atr(c)` where c=6 (default ATR period) |
| Loss Multiplier | `nLoss = a * xATR` where a=2 (key value) |
| Trailing Stop (up) | `max(prev_stop, src - nLoss)` when price above stop |
| Trailing Stop (down) | `min(prev_stop, src + nLoss)` when price below stop |
| Buy Signal | `src > trailing_stop AND ema(src,1) crosses above trailing_stop` |
| Sell Signal | `src < trailing_stop AND trailing_stop crosses above ema(src,1)` |
| Label Management | Keeps only last `maxUTLabels` (default 3) buy and sell labels |

---

## 12. HULL SUITE VARIATIONS

| Variation | Formula |
|-----------|---------|
| HMA | `wma(2 * wma(src, len/2) - wma(src, len), sqrt(len))` |
| EHMA | `ema(2 * ema(src, len/2) - ema(src, len), sqrt(len))` |
| THMA | `wma(wma(src, len/3)*3 - wma(src, len/2) - wma(src, len), len)` |

---

## 13. FVG DETECTION LOGIC

| Type | Condition |
|------|-----------|
| Bullish FVG | `low > high[2] AND close[1] > high[2] AND (low - high[2])/high[2] > threshold` |
| Bearish FVG | `high < low[2] AND close[1] < low[2] AND (low[2] - high)/high > threshold` |
| Auto Threshold | `cumulative((high-low)/low) / bar_index` |
| Bullish Mitigation | `close < fvg.min` (price drops below FVG bottom) |
| Bearish Mitigation | `close > fvg.max` (price rises above FVG top) |
| Dynamic Mode | FVG boundaries adjust: bull max shrinks toward min on close, bear min expands toward max |

---

## 14. ORDER BLOCK DETECTION LOGIC

| Component | Logic |
|-----------|-------|
| Volume Pivot | `ta.pivothigh(volume, lengthOB, lengthOB)` – volume spike detection |
| Trend State | `os = high[len] > upper ? 0 (bearish) : low[len] < lower ? 1 (bullish) : os[1]` |
| Bullish OB | Volume pivot detected while `os == 1` (bullish trend) |
| Bullish OB Zone | Top = `hl2[lengthOB]`, Bottom = `low[lengthOB]` |
| Bearish OB | Volume pivot detected while `os == 0` (bearish trend) |
| Bearish OB Zone | Top = `high[lengthOB]`, Bottom = `hl2[lengthOB]` |
| Mitigation (Wick) | Price wick penetrates OB boundary |
| Mitigation (Close) | Price closes beyond OB boundary |

---

## 15. THREE-BAR REVERSAL PATTERN LOGIC

### Bullish Pattern
| Bar | Condition |
|-----|-----------|
| Bar[2] | Bearish candle (`close[2] < open[2]`) |
| Bar[1] | Lower low AND lower high than Bar[2], bearish (`close[1] < open[1]`) |
| Bar[0] | Bullish candle (`close > open`), high exceeds Bar[2] high |
| Enhanced | Additionally: `close > high[2]` |

### Bearish Pattern
| Bar | Condition |
|-----|-----------|
| Bar[2] | Bullish candle (`close[2] > open[2]`) |
| Bar[1] | Higher high AND higher low than Bar[2], bullish (`close[1] > open[1]`) |
| Bar[0] | Bearish candle (`close < open`), low breaks below Bar[2] low |
| Enhanced | Additionally: `close < low[2]` |

### Trend Filters Available
| Filter | Method |
|--------|--------|
| Moving Average Cloud | Fast MA vs Slow MA alignment with price |
| Supertrend | Direction from `ta.supertrend(factor, atrPeriod)` |
| Donchian Channels | Highest/lowest close breakout direction |
| None | No filtering (all patterns shown) |

---

## 16. REVERSAL SIGNALS LOGIC

| Component | Logic |
|-----------|-------|
| Count Condition | `close < close[4]` (bearish counting) / inverse for bullish |
| Counter | Counts 1–9, resets to 1 after 9 |
| Bullish Momentum Signal | Counter reaches 9 (9 consecutive bars closing lower than 4 bars ago) |
| Bearish Momentum Signal | Counter reaches 9 (9 consecutive bars closing higher than 4 bars ago) |
| Resistance Level | `ta.highest(9)` at momentum completion, clears when price exceeds it |
| Support Level | `ta.lowest(9)` at momentum completion, clears when price drops below it |

---

## 17. DATA FEEDS / SECURITY CALLS

| Call | Timeframe | Data Retrieved | Purpose |
|------|-----------|---------------|---------|
| `request.security(syminfo.tickerid, "1D", ...)` | 1D | `high[1], low[1], high, low, time[1], time` | Previous day H/L, current day H/L |
| `request.security(syminfo.tickerid, htf, _hull)` | User-defined (default 240) | Hull MA value | Higher TF Hull Suite |
| `request.security(syminfo.tickerid, timeframe.period, close)` | Current TF | Close (HA adjusted) | UT Bot Heikin Ashi mode |
| `request.security(syminfo.tickerid, tfFVG, detectFVG())` | User-defined (default current) | FVG detection results | Multi-TF FVG |
| `request.security_lower_tf(syminfo.tickerid, "1", ...)` | 1-minute | OHLCV + time arrays | Lower TF rolling H/L calculation |

---

## 18. RESOURCE USAGE SUMMARY

| Resource | Limit | Used By |
|----------|-------|---------|
| Labels | 500 max | Swing labels, UT Bot, 3BR patterns, structure labels, fib labels, H/L labels |
| Lines | 500 max | Structure lines, fib levels, H/L levels, FVG mitigation, OB averages |
| Boxes | 500 max | FVG zones, OB zones, 3BR S/R zones |
| Bars Back | 500 max | Historical lookback for all calculations |
| Tables | 2 | Session dashboard (top_right), FVG dashboard (configurable) |
| `request.security` | 4 calls | 1D data, HTF Hull, HA close, MTF FVG |
| `request.security_lower_tf` | 1 call | 1-minute OHLCV for rolling TF |
